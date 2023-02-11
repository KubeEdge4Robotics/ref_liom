#include <common_lib.h>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>

#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <ikd-Tree/ikd_Tree.h>

class SubmapInfo{
public:
  double msg_time;
  int submap_index;
  PointVector cloud_ontree;
  M3D lidar_pose_rotM;
  V3D lidar_pose_tran;
  M3D corr_pose_rotM;
  V3D corr_pose_tran;
  M3D offset_R_L_I;
  M3D offset_t_L_I;
  bool oriPoseSet,corPoseSet;
  SubmapInfo():msg_time(0.0),oriPoseSet(false),corPoseSet(false){};
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
boost::shared_ptr< KD_TREE<PointType> > ikdtree_swapptr(boost::make_shared< KD_TREE<PointType> >());//用于ikdtree rebuild的临时变量
std::mutex mtx_ikdtreeptr;
std::condition_variable sig_ikdtreeptr;
bool first_correction_set = false;
bool holding_for_ikdtree_rebuild = false;
std::deque<PointCloudXYZI::Ptr> PointToAddHistorical;//需要添加到ikdtree的历史点云
PointVector PointSubmap;//子图中的点云
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr pos_kdtree;
int submap_id,last_submap_id;//订阅回环检测发生的子图id
std::deque<nav_msgs::Odometry::ConstPtr> submap_pose_buffer;//订阅回环检测发生的位姿
std::deque<nav_msgs::Path::ConstPtr> path_buffer;//存放闭环优化后的校正轨迹
std::unordered_map<int, SubmapInfo> unmap_submap_info;//索引是submap id，值存放的是子图信息
double map_retrival_range = 150;//关键帧子图最近邻搜索范围


void correctLidarPoints(PointType const * const pi, PointType * const po, const M3D &rotM, const V3D &tran){
  V3D p_ori(pi->x, pi->y, pi->z);
  V3D p_corrected(rotM*p_ori + tran);
  po->x = p_corrected(0);
  po->y = p_corrected(1);
  po->z = p_corrected(2);
  po->intensity = pi->intensity;
}

void correctLidarPoints(PointType const * const pi, PointType * const po, 
                        const M3D &R_ori, const V3D &t_ori,
                        const M3D &R_cor, const V3D &t_cor){
  V3D p_ori(pi->x, pi->y, pi->z);
  V3D p_corrected(R_cor*R_ori.transpose()*(p_ori - t_ori) + t_cor);
  po->x = p_corrected(0);
  po->y = p_corrected(1);
  po->z = p_corrected(2);
  po->intensity = pi->intensity;
}
//更新子地图的里程计位姿
bool update_submap_info(){
  std::unique_lock<std::mutex> ulock(mtx_sub_);
  if(submap_pose_buffer.empty()) return false;//子图位姿和子图的id是由loop_detection得到的
  if(submap_pose_buffer.back()->twist.covariance[0]!=submapid) return false;//submap_pose的最新值应该和submapid是对应的
  auto submap_pose_buffer_tmp = submap_pose_buffer;
  submap_pose_buffer.clear();
  //对于每个回环位姿
  for(auto &submap_pose : submap_pose_buffer_tmp){
    int idx = int(submap_pose->twist.covariance[0]);//子图id
    auto iter = unmap_submap_info.find(idx);
    if(iter != unmap_submap_info.end() && !iter->second.oriPoseSet){//如果在子图哈希表中找到idx号子图，则更新其里程计位姿
      auto &submap_info = iter->second;
      poseMsgToEigenRT(submap_pose->pose.pose, submap_info.lidar_pose_rotM,submap_info.lidar_pose_tran);
      submap_info.oriPoseSet = true;
      submap_info.msg_time = submap_pose->header.stamp.toSec();
    }
    else submap_pose_buffer.push_back(submap_pose);
  }
  return true;
}


pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr pos_kdtree, pos_kdtree_prior;//用于快速搜索的关键帧位置kdtree
pcl::PointCloud<pcl::PointXYZI>::Ptr key_poses(new pcl::PointCloud<pcl::PointXYZI>());//关键帧位置
pcl::PointCloud<pcl::PointXYZI>::Ptr key_poses_prior(new pcl::PointCloud<pcl::PointXYZI>());
std::unordered_map<int,SubmapInfo> unmap_submap_info_bkq;
//通过校正轨迹设置submap信息
void set_submap_corrected_poses(const nav_msgs::Path::ConstPtr& path_cor){
  if(!key_poses->empty()) key_poses->clear();//如果关键帧池非空，则清空
  unmap_submap_info_bkq = unmap_submap_info;//子地图信息备份
  pcl::pointXYZI posi;
  //对于每一个校正后的位姿
  for(int i=0; i < path_cor->poses.size(); i++){
    geometry_msgs::PoseStamped pose_cor = path_cor->poses[i];
    int idx = int(pose_cor.header.seq);//位姿编号
    if(idx >= 50000) idx = -(idx - 50000);//先验信息
    auto iter = unmap_submap_info.find(idx);//在子图哈希表中查找
    if(iter != unmap_submap_info.end()){
      auto &submap_info = iter->second;
      if( idx < 0 ) continue;//prior kf range
      M3D R_corr; V3D t_corr;
      poseMsgToEigenRT(pose_cor.pose, R_corr, t_corr);
      //设置子图校正后的位姿，并加入关键帧池
      submap_info.corr_pose_rotM = R_corr;
      submap_info.corr_pose_tran = t_corr;
      submap_info.corPoseSet = true;
      posi.x = submap_info.corr_pose_tran[0];
      posi.x = submap_info.corr_pose_tran[1];
      posi.x = submap_info.corr_pose_tran[2];
      posi.intensity = idx;
      key_poses->push_back(posi);
    }
  }
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr pose_kdtree_tmp(new pcl::KdTreeFLANN<pcl::PointXYZI>());//关键帧位置kdtree
  pos_kdtree_tmp->setInputCloud(key_poses);
  pos_kdtree = pos_kdtree_tmp->makeShared();//共享指针交接
}
void recover_unmap_submap_info(){
  unmap_submap_info = unmap_submap_info_bkq;
}
int last_occur_id = 0;//上一次使用校正后的submap更新ikdtree的submap的id
std::unordered_map<int,bool> submap_onikdtree_flag;
void update_ikdtree_with_submap_corrected(){
  if(!first_correction_set) return;
  if(submap_id > 1 && submap_id - last_occur_id > 0){
    last_occur_id = submap_id;
    std::vector<float> pointSearchSqDis;
    std::vector<int> pointIndices;
    pcl::PointXYZI pos_lc;//当前位姿
    pos_lc.x = kf.get_x().pose(0);
    pos_lc.y = kf.get_x().pose(1);
    pos_lc.z = kf.get_x().pose(2);
    bool use_prior = true;
    pos_kdtree->radiusSearch(pos_lc, map_retrival_range, pointIndices, pointSearchSqDis);//搜索当前位置附近的关键帧
    PointCloudXYZI::Ptr corrected_cloud_submap_local(new PointCloudXYZI());//校正后的局部子图点云
    //对每一个近邻关键帧
    for(auto idx:pointIndices){
      int kp_index;//关键帧编号
      kp_index = key_poses->points[idx].intensity;
      if (submap_onikdtree_flag[kp_index] == true) continue;//如果该关键帧的点云已经在ikd_tree上了，则跳过当前循环
      
      auto iter = unmap_submap_info.find(kp_index);
      if(iter == unmap_submap_info.end()) continue;//如果在子图hash表中没有找到当前关键帧的子图则跳过当前循环
      auto &submap_info = iter->second;
      if(!submap_info.corPoseSet || !submap_info.oriPoseSet) continue;//如果该关键帧子图没有位姿校正则跳过当前循环
      int cloud_size = submap_info.cloud_ontree.size();
      submap_onikdtree_flag[kp_index] = true;//将子图在ikdtree上的标志位设置为true
      PointCloudXYZI::Ptr laserCloudWorldCorrected(new PointCloudXYZI(cloud_size, 1));
      if(kp_index < 0){//是先验信息，直接加入
        for (int i = 0; i < cloud_size; i++)
        {
          laserCloudWorldCorrected->push_back(submap_info.cloud_ontree[i]);
        }
      }
      else{//校正子图点云
        for (int i = 0; i < cloud_size; i++)
        {
          correctLidarPoints(&submap_info.cloud_ontree[i], &laserCloudWorldCorrected->points[i],\
                              submap_info.lidar_pose_rotM, submap_info.lidar_pose_tran,\
                              submap_info.corr_pose_rotM, submap_info.corr_pose_tran );
        }
      }
      *corrected_cloud_submap_local += *laserCloudWorldCorrected;
    }
    PointCloudXYZI::Ptr cloud_ds(new PointCloudXYZI());//校正后地图点云经过降采样后的点云
    pcl::VoxelGrid<PointType> sor;
    float leafsize = 0.75;
    sor.setLeafSize(leafsize, leafsize, leafsize);
    sor.setInputCloud(corrected_cloud_submap_local);
    sor.filter(cloud_ds);
    PointToAddHistorical.push_back(cloud_ds);
  }
}
vector<PointVector> PointsNearVec(100000);//近邻点向量
float point_dis[100000] = {0,0};
PointCloudXYZI::Ptr point_world_guess(new PointCloudXYZI());//icp源点云
PointCloudXYZI::Ptr point_near_;//源点云最近点
//通过点到点icp优化
bool refine_with_pt2pt_icp(const PointCloudXYZI::Ptr &feats_down_body, const float rmse_thr, const int iteration_thr,\
                           const boost::shared_ptr<KD_TREE<PointType>> &ikd_in,M3D &R_icp, V3D &t_icp, const M3D &R_guess,
                           const V3D &t_guess, const int pos_id_lc){
    int cloud_size = feats_down_body->size();
    R_icp = M3D::Identity(); t_icp = V3D(0,0,0);
    MD(4,4) T_icp = MD(4,4)::Identity();//icp结果转换矩阵
    float rmse = 0.0f;//方均根误差
    int iteration = 0;//迭代次数
    //开始icp迭代
    while(iteration < iteration_thr){
      point_world_tmp->clear();
      point_near_tmp->clear();
      int num_match = 1;
      #if _OPENMP
      #pragma omp parallel for
      #endif
      for(int i = 0; i<cloud_size; i++){
        const PointType point_body = feats_down_body->points[i];
        V3D p_global(R_icp * (R_guess * p_body + t_guess) + t_icp);
        PointType &point_world = feats_down_world_guess->points[i];
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;
        PointVector &points_near = PointsNearVec[i];
        vector<float> pointSearchSqDis(num_match);
        ikd_in->Nearest_Search(point_world, num_match, points_near, pointSearchSqDis);
        point_dis[i] = sqrt(pointSearchSqDis[0]);
      }
      for (int i = 0; i < cloud_size; i++)
      {
        rmse_sum += point_dis[i];
        if(point_dis[i] < rmse_thr) continue;
        point_world_tmp->push_back(point_world);
        point_near_tmp->push_back(point_near);
      }
      rmse = rmse_sum/float(cloud_size);
      if(rmse < rmse_thr){
        return true;
      }
      int new_size = point_near_tmp->size();
      Eigen::Matrix<double, 3, Eigen::Dynamic> point_src(3, new_size);
      Eigen::Matrix<double, 3, Eigen::Dynamic> point_tgt(3, new_size);
      for(int i = 0; i < new_size; i++){
        PointType &point_near = point_near_tmp->points[i];
        PointType &point_world = point_world_tmp->points[i];
        point_src(0, i) = point_world.x;
        point_src(1, i) = point_world.y;
        point_src(2, i) = point_world.z;
        point_src(0, i) = point_near.x;
        point_tgt(1, i) = point_near.y;
        point_tgt(2, i) = point_near.z;
      }
      MD(4,4) T_delta = pcl::umeyama(point_src, point_tgt, false);//SVD分解求解icp问题
      T_icp = T_delta * T_icp;
      R_icp = T_icp.block<3,3>(0,0);
      t_icp = T_icp.block<3,1>(0,3);
      
      iteration++;
    }
    if(rmse < rmse_thr) return true;
    return false;
}

void set_KF_pose(esekfom::esekf<state_ikfom, 12, input_ikfom> &kf, 
                 state_ikfom &tmp_state, 
                 const MD(4,4) &Tcomb_nooff,
                 const M3D &Rvel
){
  state_ikfom 
  tmp_state.rot = (Ticp*Tcomb_nooff).block<3,3>(0,0);
  tmp_state.pos = (Ticp*Tcomb_nooff).block<3,1>(0,3);
  tmp_state.vel = Rvel * tmp_state.vel;
  kf.change_x(tmp_state);
}

V3D last_update_pose(0,0,0);
int corrected_counter = 0, ikdtreebuild_counter = 0;
int corrected_submapsize = 0;  
void ikdtree_rebuild(){
  ros::Rate rate(20);
  while(ros::ok()){
    rate.sleep();
    sig_ikdtreeptr.notify_all();//allow fastlio thread do its things
    if(last_submap_id == submap_id) continue;//no new kf, do nothing
    update_submap_info();//更新子地图的里程计位姿
    update_ikdtree_with_submap_corrected();//将当前位姿周围的校正子图加入ikdtree中
    if(path_buffer.empty() || unmap_submap_info.empty()) continue;//如果校正轨迹为空或历史子图信息为空则跳过此次循环 
    V3D curr_update_pose(path_buffer.front()->poses.back().pose.position.x,
                        path_buffer.front()->poses.back().pose.position.y,\
                        path_buffer.front()->poses.back().pose.position.z);//最新的闭环校正位置
    float update_dis = (curr_update_pose - last_update_pose).norm();//当前校正位置和上一次校正位置如果小于阈值则跳过当前循环
    if(update_dis < correction_dis_interval && first_correction_set){
      path_buffer.pop_front();
      continue;
    }
    set_submap_corrected_poses(path_buffer.front());//通过校正轨迹设置submap信息
    path_buffer.pop_front();

    int submap_size = unmap_submap_info.size();//submap哈希表的元素个数
    std::vector<float> pointSearchSqDis;
    std::vector<int> pointIndices;
    pcl::PointXYZI pos_lc;
    int pos_id_lc = submap_size - 1;
    while (!unmap_submap_info[pos_id_lc].corPoseSet)
    {//遍历submap哈希表，对位姿校正后的submap地图进行处理
      pos_id_lc--;
    }
    //回环校正后的子图位置
    pos_lc.x = unmap_submap_info[pos_id_lc].corr_pose_tran[0];
    pos_lc.y = unmap_submap_info[pos_id_lc].corr_pose_tran[1];
    pos_lc.z = unmap_submap_info[pos_id_lc].corr_pose_tran[2];
    pos_kdtree->radiusSearch(pos_lc, map_retrival_range, pointIndices, pointSearchSqDis);//搜索回环附近的关键帧位置
    std::vector<int> submap_ids_vec;
    PointCloudXYZI::Ptr corrected_cloud_submap_local(new PointCloudXYZI());//校正后的局部子图点云
    int kf_index;
    //对于回环的每个相邻关键帧位置，校正其点云
    for(auto pidx : pointIndices){
      kf_index = key_poses->points[pidx].intensity;//关键帧位姿编号      
      auto iter = unmap_submap_info.find(kf_index);
      if(iter == unmap_submap_info.end()) continue;//如果在子图哈希表中找不到则跳过执行下一循环

      auto &submap_info = iter->second;
      if(!submap_info.corPoseSet || !submap_info.oriPoseSet) continue;//如果子图位姿未校正则跳过执行下一循环

      int cloud_size = submap_info.cloud_ontree.size();
      if(cloud_size == 0) continue;
      submap_ids_vec.push_back(kf_index);//submap_onikdtree_flag[kf_index]=true
      PointCloudXYZI::Ptr laserCloudWorldCorrected(new PointCloudXYZI(cloud_size, 1));//该帧校正后的世界系点云
      if(kf_index < 0){//如果是先验，直接加入
        for (int i = 0; i < cloud_size; i++)
        {
          laserCloudWorldCorrected->push_back(submap_info.cloud_ontree[i]);
        }
      }
      else{//否则用校正后位姿修正每个点
        for (int i = 0; i < cloud_size; i++)
        {
          correctLidarPoints(&submap_info.cloud_ontree[i], &laserCloudWorldCorrected->points[i],\
                              submap_info.lidar_pose_rotM, submap_info.lidar_pose_tran,\
                              submap_info.corr_pose_rotM, submap_info.corr_pose_tran );
        }
      }
      *corrected_cloud_submap_local += *laserCloudWorldCorrected;//将每帧点云汇总
    }
    if(corrected_cloud_submap_local->empty()) continue;
    if(corrected_cloud_submap_local->size() > 150000){//如果校正后局部地图点云过大，则降采样
      int num_pt_cur = corrected_cloud_submap_local->size();
      std::vector<int> indices;
      int sample_gap = ceil(double(num_pt_cur)/double(200000));
      for(int i=0; i < corrected_cloud_submap_local->size(); i+=sample_gap ){
        indices.push_back(i);
      }
      PointCloudXYZI::Ptr corrected_cloud_submap_tmp(new PointCloudXYZI());
      pcl::copyPointCloud(*corrected_cloud_submap_local, indices, *corrected_cloud_submap_tmp);
      corrected_cloud_submap_local = corrected_cloud_submap_tmp;//降采样
    }
    corrected_submapsize += corrected_cloud_submap_local->size();//校正点数量更新
    corrected_counter++;//闭环校正计数加1

    //重建新ikdtree，之后将ikdtree_swapptr和ikdtree_ptr互换
    if(ikdtree_swapptr->Root_Node == nullptr)
    {
        ikdtree_ptr->set_downsample_param(filter_size_map_min);//设置ikd_tree的降采样参数
        ikdtree_ptr->Build(corrected_cloud_submap_local->points);//从零构建ikd树
        ikdtreebuild_counter++;//ikdtree重建次数计数
    }

    bool correction_succeed = true;
    
    M3D R_ori, R_cor, R_pre;
    V3D t_ori, t_cor, t_pre;
    MD(4,4) T_ori, T_cor, T_pre;
    T_pre = Rt2T(tmp_state.rot.toRotationMatrix(), tmp_state.pos);//imu的状态
    {
      unique_lock<std::mutex> my_unique_lock(mtx_ikdtreeptr);
      holding_for_ikdtree_rebuild = true;//设置条件，阻塞其他线程
    }
    if(unmap_submap_info[pos_id_lc].oriPoseSet && unmap_submap_info[pos_id_lc].corPoseSet){
      T_ori = Rt2T(unmap_submap_info[pos_id_lc].lidar_pose_rotM, unmap_submap_info[pos_id_lc].lidar_pose_tran);//原始里程计位姿
      T_cor = Rt2T(unmap_submap_info[pos_id_lc].corr_pose_rotM, unmap_submap_info[pos_id_lc].corr_pose_tran);//闭环校正位姿
    }
    MD(4,4) T_off = Rt2T(tmp_state.offset_R_L_I.toRotationMatrix(),tmp_state.offset_T_L_I);//lidar到imu的转换矩阵
    MD(4,4) T_guess = T_cor * T_ori.inverse() * T_pre * T_off;//imu状态的预测
    MD(4,4) T_icp = MD(4,4)::Identity();//icp优化结果
    if (!refine_with_pt2pt_icp(feats_down_body, correction_ver_thr, 10, ikdtree_swapptr, T_icp, T_guess)){
      {
        unique_lock<std::mutex> my_unique_lock(mtx_ikdtreeptr);
        holding_for_ikdtree_rebuild = false;
      }
      sig_ikdtreeptr.notify_all();
      corrected_cloud_submap_local->clear();
      ikdtree_swapptr = boost::make_shared<KD_TREE<PointType>>();
      recover_unmap_submap_info();
      correction_succeed = false;
      std::cout << "bad correction" <<std::endl;
    }
    else{
      MD(4,4) T_icp = Rt2T(R_icp, t_icp);
      M3D Rvel = R_icp * R_cor * R_ori.transpose();
      set_KF_pose(kf, guess, Ticp, guess*Toff.inverse(),Rvel);
      std::swap(ikdtree_ptr, ikdtree_swapptr);
      unique_lock<std::mutex> my_unique_lock(mtx_ikdtreeptr);
      holding_for_ikdtree_rebuild = false;
      sig_ikdtreeptr.notify_all();
    }
      
    if(!correction_succeed) continue;
    ikdtree_swapptr = boost::make_shared<KD_TREE<PointType>>();
    submap_onikdtree_flag.clear();//reset flag when new pose correction done
    for (auto idx:submap_ids_vec)
    {
      submap_onikdtree_flag[idx] = true;
    }
    last_update_pose = curr_update_pose;
    first_correction_set = true;
  }
}