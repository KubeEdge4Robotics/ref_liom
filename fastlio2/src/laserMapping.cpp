// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <unordered_map>//哈希表
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include "cmake_config.h"
#ifdef NODELET
#include "laserMapping.h"
#endif

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)

/*** Time Log Variables ***/
// kdtree_incremental_time为kdtree建立时间，kdtree_search_time为kdtree搜索时间，kdtree_delete_time为kdtree删除时间
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
// T1为雷达初始时间戳，s_plot为整个流程耗时，s_plot2特征点数量,s_plot3为kdtree增量时间，s_plot4为kdtree搜索耗时，s_plot5为kdtree删除点数量
//，s_plot6为kdtree删除耗时，s_plot7为kdtree初始大小，s_plot8为kdtree结束大小,s_plot9为平均消耗时间，s_plot10为添加点数量，s_plot11为点云预处理的总时间
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
// 定义全局变量，用于记录时间,match_time为匹配时间，solve_time为求解时间，solve_const_H_time为求解H矩阵时间
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
// kdtree_size_st为ikd-tree获得的节点数，kdtree_size_end为ikd-tree结束时的节点数，add_point_size为添加点的数量，kdtree_delete_counter为删除点的数量
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
// runtime_pos_log运行时的log是否开启，pcd_save_en是否保存pcd文件，time_sync_en是否同步时间
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};             //残差，点到面距离平方
float DET_RANGE = 300.0f;                   //雷达探测范围
const float MOV_THRESHOLD = 1.5f;           //雷达中心到地图边缘面与雷达探测范围的比例
double time_diff_lidar_to_imu = 0.0;        //Time offset between lidar and IMU

string root_dir = ROOT_DIR;                 //设置根目录
string map_file_path, lid_topic, imu_topic; //设置地图文件路径，雷达topic，imu topic
string odom_frame, body_frame;              //fastlio世界坐标系, IMU坐标系

double res_mean_last = 0.05, total_residual = 0.0;                  //设置残差平均值，残差总和
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;         //最新雷达时间戳，imu时间戳
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;//设置imu的角速度协方差，加速度协方差，角速度协方差偏置，加速度协方差偏置
double filter_size_single_frame = 0;//单帧激光雷达数据voxel滤波的最小尺寸
double filter_size_map_ikdtree = 0;//ikdtree地图的最小尺寸
double fov_deg = 0;//雷达视野角度
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_reset, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<vector<int>>  pointSearchInd_surf;       //每个点的索引,暂时没用到
vector<BoxPointType> cub_needrm;                // ikd-tree中，地图需要移除的包围盒序列
vector<PointVector>  Nearest_Points;            //每个点的最近点序列
vector<double>       extrinT(3, 0.0);           //雷达相对于IMU的偏移
vector<double>       extrinR(9, 0.0);           //雷达相对于IMU的旋转矩阵
deque<double>                     time_buffer;  // 激光雷达数据时间戳缓存队列
deque<PointCloudXYZI::Ptr>        lidar_buffer; //记录特征提取或间隔采样后的lidar（特征）数据
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;   // IMU数据缓存队列

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());             //ikd-tree中的点云
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());          //去畸变的雷达点云
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());          //畸变校正后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());         //畸变校正后降采样的单帧点云，世界系
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));         //特征点在地图中的对应点的局部平面参数，世界系
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));   //lidar系的有效点点云
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));   //对应点法向量
PointCloudXYZI::Ptr _featsArray;                                    //ikd-tree中需要移除的点云

pcl::VoxelGrid<PointType> downSizeFilterSingleFrame;   //单帧内降采样使用voxel grid

boost::shared_ptr< KD_TREE<PointType> > ikdtree_ptr(boost::make_shared< KD_TREE<PointType> >());

//ikdtree rebuild
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
//ikdtree rebuild

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;//esekf实例，<状态，噪声维度，输入>
state_ikfom kf_state;                           //Kalman filter状态
vect3 pos_lid;                                  //世界系下lidar坐标
//发布的路径参数
nav_msgs::Path::Ptr path(new nav_msgs::Path);   //轨迹，包含一系列位姿

geometry_msgs::Quaternion geoQuat;              //四元数
geometry_msgs::PoseStamped msg_body_pose;       //位姿
//激光雷达和imu处理操作
shared_ptr<Preprocess> p_pre(new Preprocess()); //定义指向激光雷达数据的预处理类Preprocess的智能指针
shared_ptr<ImuProcess> p_imu(new ImuProcess()); //定义指向IMU数据预处理类ImuProcess的智能指针

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(kf_state.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", kf_state.pos(0), kf_state.pos(1), kf_state.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", kf_state.vel(0), kf_state.vel(1), kf_state.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", kf_state.bg(0), kf_state.bg(1), kf_state.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", kf_state.ba(0), kf_state.ba(1), kf_state.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", kf_state.grav[0], kf_state.grav[1], kf_state.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(kf_state.rot * (kf_state.offset_R_L_I*p_body + kf_state.offset_T_L_I) + kf_state.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(kf_state.rot * (kf_state.offset_R_L_I*p_body + kf_state.offset_T_L_I) + kf_state.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(kf_state.rot * (kf_state.offset_R_L_I*p_body + kf_state.offset_T_L_I) + kf_state.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(kf_state.offset_R_L_I*p_body_lidar + kf_state.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}
//得到被剔除的点
void points_cache_collect()
{
    PointVector points_history;
    ikdtree_ptr->acquire_removed_points(points_history);//返回被剔除的点
    for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);//存入到缓存中，后续没有用到该数据
}
// 动态调整地图区域，防止地图过大而内存溢出，类似LOAM中提取局部地图的方法
BoxPointType LocalMap_Points;// ikd-tree中,局部地图的包围盒角点
bool Localmap_Initialized = false;// 局部地图是否初始化
void lasermap_fov_segment()
{
    cub_needrm.clear(); //清空需要清除的区域队列
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);//X轴分界点转换到世界系下，后续没有用到
    V3D pos_LiD = pos_lid;//世界系下lidar的位置
    if (!Localmap_Initialized){ //初始化局部地图包围盒角点,以世界系下的lidar位置为中心，cube_len为边长的立方体
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];//各个方向上Lidar与局部地图边界的距离，或者说是lidar与立方体盒子六个面的距离
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        //与某个方向上的边界距离（例如1.5*300m）太小，标记需要移除need_move，参考论文Fig3
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;//如果不需要移除，即探测区域没有超出地图范围则返回
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;//新的局部地图包围盒边界点
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));//需要移动的距离
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){//与包围盒面的距离小于阈值
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;//沿该方向画出需要移除的局部地图
            cub_needrm.push_back(tmp_boxpoints);//将需要移除的区域加入需要清除的队列
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree_ptr->Delete_Point_Boxes(cub_needrm);//使用包围盒删除指定区域内的点
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}
//除livox雷达之外的雷达点云的回调函数，将数据加入到buffer中
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();//记录时间
    double  msg_header_stamp = msg->header.stamp.toSec();
    if(p_pre->lidar_type == RS32){
        sensor_msgs::PointCloud2ConstIterator<double> iter_time(*msg,"timestamp");
        msg_header_stamp = *iter_time;
    }
    if (msg_header_stamp < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);//点云预处理
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg_header_stamp);
    last_timestamp_lidar = msg_header_stamp;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);

    double timestamp = msg->header.stamp.toSec();

    if (timestamp < last_timestamp_imu)
    {
        flg_reset = true;
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
//处理buffer中的数据，将两帧激光雷达点云数据时间内的IMU数据从缓存队列中取出，进行时间对齐，并保存带meas中
bool sync_packages(MeasureGroup &meas)
{
    // cout<<"sync_measurements"<<endl;
    if (lidar_buffer.empty() || imu_buffer.empty()) {//如果缓存队列中没有数据，则返回false
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)//如果还没有把雷达数据放入meas中的话，就执行压入雷达数据
    {
        meas.lidar = lidar_buffer.front();//从激光雷达点云缓存队列中取出点云数据，放到meas中
        meas.lidar_beg_time = time_buffer.front();//该帧lidar测量的起始时间
        if (meas.lidar->points.size() <= 1) // 该帧雷达点云没有点，返回false
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;//此帧点云结束时刻

        lidar_pushed = true;//成功提取到雷达测量的标志
    }
    // 最新的IMU时间戳(也就是队尾的)不能早于雷达的end时间戳，则返回false
    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    //拿出lidar_beg_time到lidar_end_time之间的所有IMU数据
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();//获取imu数据的时间戳
        if(imu_time > lidar_end_time) break;//如果时间大于lidar_end_time了则跳出循环
        meas.imu.push_back(imu_buffer.front());//将imu数据放到meas中
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();   //将lidar数据弹出
    time_buffer.pop_front();    //将时间戳弹出
    lidar_pushed = false;       //将lidar_pushed置为false，代表lidar数据对应的imu数据已经被放到meas中了
    return true;
}

int process_increments = 0;
void map_incremental()//地图的增量更新，主要完成对ikd-tree的地图建立
{
    PointVector PointToAdd;                         //需要加入到ikd-tree中的点云
    PointVector PointNoNeedDownsample;              //加入ikd-tree时，不需要降采样的点云
    PointToAdd.reserve(feats_down_size);            // 提前申请内存
    PointNoNeedDownsample.reserve(feats_down_size); 
    //根据点与所在包围盒中心的距离，分类是否需要降采样
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        //判断是否有关键点需要加到地图中
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i]; //获取附近的点云
            bool need_add = true;                               //是否需要加到地图中
            BoxPointType Box_of_Point;                          //点云所在的包围盒
            PointType downsample_result, mid_point;             //降采样结果，中点
            //filter_size_map_min是地图体素降采样的栅格边长
            //mid_point即为该特征点所属的体素的中心点坐标
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_ikdtree)*filter_size_map_ikdtree + 0.5 * filter_size_map_ikdtree;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_ikdtree)*filter_size_map_ikdtree + 0.5 * filter_size_map_ikdtree;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_ikdtree)*filter_size_map_ikdtree + 0.5 * filter_size_map_ikdtree;
            float dist  = calc_dist(feats_down_world->points[i],mid_point); //计算点与体素中心的距离
            //判断最近点在x、y、z三个方向上，与体素中心的距离，判断加入时是否需要降采样
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_ikdtree && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_ikdtree && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_ikdtree){
                //如果三个方向距离都大于地图栅格半轴长，则无需降采样
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            //判断当前点的NUM_MATCH_POINTS个邻近点与包围盒中心的范围
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;//若邻近点数小于NUM_MATCH_POINTS，则直接跳出
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {//如果存在邻近点到中心的距离小于当前点到中心的距离，则不需要添加当前点
                    need_add = false;
                    break;
                }
            }
            if (need_add){
                PointToAdd.push_back(feats_down_world->points[i]);//如果需要添加则将点加入到PointToAdd中
                // if (PointToAdd.size()%3 == 0) PointSubmap.push_back(feats_down_world->points[i]);//降采样加入PointSubmap中
            }
        }
        else
        {//如果周围没有点或者没有初始化EKF，则加入到PointToAdd中
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();       //记录起始时间
    PointVector PointToAddHistoricalFront;
    if(!PointToAddHistorical.empty()){
        PointToAddHistoricalFront = PointToAddHistorical.front()->points;
        PointToAddHistorical.pop_front();
    }
    {
        unique_lock<std::mutex> my_unique_lock(mtx_ikdtreeptr);
        sig_ikdtreeptr.wait(my_unique_lock, []{return !holding_for_ikdtree_rebuild;});
        add_point_size = ikdtree_ptr->Add_Points(PointToAdd, true); //加入点时需要降采样
        ikdtree_ptr->Add_Points(PointNoNeedDownsample, false);      //加入点时不需要降采样
        if(!PointToAddHistoricalFront.empty()){
            add_point_size = ikdtree_ptr->Add_Points(PointToAddHistoricalFront, true);
        }
        sig_ikdtreeptr.notify_all();
    }
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();  //计算总共加入ikd-tree的点的数量
    kdtree_incremental_time = omp_get_wtime() - st_time;                //kdtree增长耗时
}

// PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2Ptr laserCloudmsg(new sensor_msgs::PointCloud2());
        pcl::toROSMsg(*laserCloudWorld, *laserCloudmsg);
        laserCloudmsg->header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg->header.frame_id = odom_frame;

        pubLaserCloudFull.publish(laserCloudmsg);
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }
    
    sensor_msgs::PointCloud2Ptr laserCloudmsg(new sensor_msgs::PointCloud2());
    pcl::toROSMsg(*laserCloudIMUBody, *laserCloudmsg);
    laserCloudmsg->header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg->header.frame_id = body_frame;
    pubLaserCloudFull_body.publish(laserCloudmsg);
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }

    sensor_msgs::PointCloud2Ptr laserCloudFullRes3(new sensor_msgs::PointCloud2());
    pcl::toROSMsg(*laserCloudWorld, *laserCloudFullRes3);
    laserCloudFullRes3->header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3->header.frame_id = odom_frame;
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2Ptr laserCloudMap(new sensor_msgs::PointCloud2());
    pcl::toROSMsg(*featsFromMap, *laserCloudMap);
    laserCloudMap->header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap->header.frame_id = odom_frame;
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = kf_state.pos(0);
    out.pose.position.y = kf_state.pos(1);
    out.pose.position.z = kf_state.pos(2);
    out.pose.orientation.x = kf_state.rot.coeffs()[0];
    out.pose.orientation.y = kf_state.rot.coeffs()[1];
    out.pose.orientation.z = kf_state.rot.coeffs()[2];
    out.pose.orientation.w = kf_state.rot.coeffs()[3];
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    
    nav_msgs::Odometry::Ptr odomAftMapped(new nav_msgs::Odometry);//一个位姿
    odomAftMapped->header.frame_id = odom_frame;
    odomAftMapped->child_frame_id = body_frame;
    odomAftMapped->header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped->pose);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped->pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped->pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped->pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped->pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped->pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped->pose.covariance[i*6 + 5] = P(k, 2);
    }
    pubOdomAftMapped.publish(odomAftMapped); 
    
    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped->pose.pose.position.x, \
                                    odomAftMapped->pose.pose.position.y, \
                                    odomAftMapped->pose.pose.position.z));
    q.setW(odomAftMapped->pose.pose.orientation.w);
    q.setX(odomAftMapped->pose.pose.orientation.x);
    q.setY(odomAftMapped->pose.pose.orientation.y);
    q.setZ(odomAftMapped->pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped->header.stamp, odom_frame, body_frame ) );
    
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = odom_frame;

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path->poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}
//计算残差信息
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #if _OPENMP
    #pragma omp parallel for
    #endif
    //对每个降采样后的特征点寻找最近邻面
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);//lidar系点云
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos); //lidar系点云转换到世界系下
        //赋值给世界系下点云
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)//如果eskf收敛了
        {
            /** Find the closest surfaces in the map **/
            ikdtree_ptr->Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            //如果最近邻的点数小于NUM_MATCH_POINTS或者最近邻点到特征点的距离大于5m，则认为该点不是有效点
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;//如果该店不是有效点，则执行下一循环

        VF(4) pabcd;//平面点信息
        point_selected_surf[i] = false;//将该点设置为无效点
        //拟合平面方程a*x+b*y+c*z+d=0并求解点到平面的距离
        if (esti_plane(pabcd, points_near, 0.1f))//找平面点法向量寻找，common_lib.h中的函数
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);//计算点到平面的距离
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)//如果残差大于阈值，则认为该点是有效点
            {
                point_selected_surf[i] = true;//再次恢复为有效点
                normvec->points[i].x = pabcd(0);//将法向量存储到normvec
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;//将点到平面的距离存储值normvec的intensity中
                res_last[i] = abs(pd2);//将残差存储到res_last
            }
        }
    }
    
    effct_feat_num = 0;//有效特征点数

    for (int i = 0; i < feats_down_size; i++)
    {//根据point_selected_surf状态判断那些点是可用的
        if (point_selected_surf[i])
        {   
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];//lidar系有效点存到laserCloudOri中
            corr_normvect->points[effct_feat_num] = normvec->points[i];//法向量存到corr_normvect中
            total_residual += res_last[i]; //残差求和
            effct_feat_num ++;//有效点数加1
        }
    }
    //如果有效点数过少，则返回
    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;//求平均残差
    match_time  += omp_get_wtime() - match_start;//匹配消耗时间
    double solve_start_  = omp_get_wtime();// 求解开始的时间
    
    /*** 观测Jacobian矩阵H=J*P*J' 及 测量向量的计算 ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //测量雅可比矩阵H，对应论文式23
    ekfom_data.h.resize(effct_feat_num);//测量向量h
    //求观测值与误差的Jacobian矩阵，论文式14及12、13
    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];//拿到有效点lidar系坐标
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;//计算点反对称矩阵
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;//转换到IMU系下
        M3D point_crossmat;//IMU系下坐标的反对称矩阵
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];//点的法向量
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);//R^{-1} * 法向量，s.rot.conjugate()是四元数共轭，即旋转矩阵求逆
        V3D A(point_crossmat * C);//法向量是在世界系下
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

#ifdef NODELET
int mainLIOFunction()
{
    int argc; char** argv;
    ros::init(argc, argv, "laserMapping");
#else
int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
#endif
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en",path_en, true);                                    // 是否发布轨迹
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);                        // 是否发布当前正在扫描的点云的topic
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);                      // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic，
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);             // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic，需要该变量和上一个变量同时为true才发布
    
    nh.param<string>("common/odom_frame",odom_frame,"camera_init");                     // fastlio世界坐标系
    nh.param<string>("common/body_frame",body_frame,"body");                            // IMU坐标系
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");                      // 激光雷达点云topic名称
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");                       // IMU的topic名称
    nh.param<bool>("common/time_sync_en", time_sync_en, false);                         // 是否需要时间同步，只有当外部未进行时间同步时设为true
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);   // lidar相对imu的时间偏置
    
    nh.param<int>("mapping/max_iteration",NUM_MAX_ITERATIONS,4);                        // 卡尔曼滤波的最大迭代次数
    nh.param<double>("mapping/filter_size_single_frame",filter_size_single_frame,0.5);  // 一帧雷达点云VoxelGrid降采样时的体素大小
    nh.param<double>("mapping/filter_size_map_ikdtree",filter_size_map_ikdtree,0.5);    // 局部地图ikdtree降采样时的体素大小
    nh.param<double>("mapping/cube_side_length",cube_len,200);                          // 局部地图立方体的边长
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);                               // 激光雷达的最大探测范围
    nh.param<double>("mapping/fov_degree",fov_deg,180);                                 // 激光雷达的视场角
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);                                    // IMU陀螺仪的协方差
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);                                    // IMU加速度计的协方差
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);                             // IMU陀螺仪偏置的协方差
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);                             // IMU加速度计偏置的协方差
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());         // 雷达相对于IMU的外参T（即雷达在IMU坐标系中的坐标）
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());         // 雷达相对于IMU的外参R
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);                 // 是否估计外参
    
    nh.param<bool>("preprocess/feature_extract_enable", p_pre->feature_enabled, false); // 是否提取特征点（FAST_LIO2默认不进行特征点提取）
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);                           // 最小距离阈值，即过滤掉0～blind范围内的点云
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);                    // 激光雷达的类型
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);                          // 激光雷达扫描的线数
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);                   // lidar数据时间单位
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);                        // lidar扫描频率
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 2);           // 采样间隔，即每point_filter_num个点取1个点
   
    nh.param<bool>("pcd_save/runtime_pos_log_enable", runtime_pos_log, 0);                       // 是否输出调试log信息
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);                         // 是否保存pcd文件
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<string>("pcd_save/map_file_path",map_file_path,"");                         // 地图保存路径

    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    cout<<"scan_pub_en "<<scan_pub_en<<endl;
    cout<<"scan_bodyframe_pub_en "<<scan_body_pub_en<<endl;

    
    path->header.stamp    = ros::Time::now();
    path->header.frame_id =odom_frame;

    /*** variables definition ***/
        /** 变量定义
     * effect_feat_num          （后面的代码中没有用到该变量）
     * frame_num                雷达总帧数
     * deltaT                   （后面的代码中没有用到该变量）
     * deltaR                   （后面的代码中没有用到该变量）
     * aver_time_consu          每帧平均的处理总时间
     * aver_time_icp            每帧中icp的平均时间
     * aver_time_match          每帧中匹配的平均时间
     * aver_time_incre          每帧中ikd-tree增量处理的平均时间
     * aver_time_solve          每帧中计算的平均时间
     * aver_time_const_H_time   每帧中计算的平均时间（当H恒定时）
     * flg_EKF_converged        （后面的代码中没有用到该变量）
     * EKF_stop_flg             （后面的代码中没有用到该变量）
     * FOV_DEG                  （后面的代码中没有用到该变量）
     * HALF_FOV_COS             （后面的代码中没有用到该变量）
     * _featsArray              （后面的代码中没有用到该变量）
     */
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * M_PI / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    downSizeFilterSingleFrame.setLeafSize(filter_size_single_frame, filter_size_single_frame, filter_size_single_frame);// VoxelGrid滤波器参数，即进行滤波时的创建的体素边长为filter_size_surf_min
    memset(point_selected_surf, true, sizeof(point_selected_surf));// 将数组point_selected_surf内元素的值全部设为true，数组point_selected_surf用于选择平面点
    memset(res_last, -1000.0f, sizeof(res_last));// 将数组res_last内元素的值全部设置为-1000.0f，数组res_last为残差数组

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    // 设置IMU的参数，对p_imu进行初始化，其中p_imu为ImuProcess的智能指针（ImuProcess是进行IMU处理的类）
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    // 将函数地址传入kf对象中，用于接收特定于系统的模型及其差异
    // 作为一个维数变化的特征矩阵进行测量。
    // 通过一个函数（h_dyn_share_in）同时计算测量（z）、估计测量（h）、偏微分矩阵（h_x，h_v）和噪声协方差（R）。
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);// 雷达点云的订阅器sub_pcl，订阅点云的topic
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);// IMU的订阅器sub_imu，订阅IMU的topic
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);// 发布当前正在扫描的点云，topic名字为/cloud_registered
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);// 发布经过运动畸变校正到IMU坐标系的点云，topic名字为/cloud_registered_body
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path>("/path", 100000);
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);// 中断处理函数，如果有中断信号（比如Ctrl+C），则执行第二个参数里面的SigHandle函数
    ros::Rate rate(1000);
    while (ros::ok())
    {
        if (flg_exit) break;
        ros::spinOnce();
        if(sync_packages(Measures)) // 将激光雷达点云数据和IMU数据从缓存队列中取出，进行时间对齐，并保存到Measures中
        {
            // 激光雷达第一次扫描
            if(flg_reset){
                ROS_WARN("reset when rosbag play back");
                p_imu.reset();
                Measures.imu.clear();//把imu数据清空
                flg_reset = false;
                continue;
            }
            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();
            {
                unique_lock<std::mutex> my_unique_lock(mtx_ikdtreeptr);
                sig_ikdtreeptr.wait(my_unique_lock, []{return !holding_for_ikdtree_rebuild;});
                p_imu->Process(Measures, kf, feats_undistort);//对IMU数据进行预处理，其中包含了点云畸变处理 前向传播 反向传播
                sig_ikdtreeptr.notify_all();
            }
            kf_state = kf.get_x();//获取kf预测的状态
            pos_lid = kf_state.pos + kf_state.rot * kf_state.offset_T_L_I;//世界坐标系下雷达的位置

            if (feats_undistort->empty() || (feats_undistort == NULL))//如果去畸变点云数据为空，则说明雷达没有完成去畸变，此时还不能初始化成功
            {
                first_lidar_time = Measures.lidar_beg_time;//记录第一次扫描的时间
                p_imu->first_lidar_time = first_lidar_time;//将第一帧时间传给imu
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            //判断EKF初始化是否成功
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
            /*** Segment the map in lidar FOV 动态调整局部地图，在拿到eskf前馈结果后***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSingleFrame.setInputCloud(feats_undistort);//获取去畸变后的点云数据
            downSizeFilterSingleFrame.filter(*feats_down_body);//滤波降采样后的点云数据
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();//记录滤波后的点云数量
            /*** initialize the map kdtree ***/
            if(ikdtree_ptr->Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    unique_lock<std::mutex> my_unique_lock(mtx_ikdtreeptr);
                    sig_ikdtreeptr.wait(my_unique_lock, []{return !holding_for_ikdtree_rebuild;});
                    ikdtree_ptr->set_downsample_param(filter_size_map_ikdtree);//设置ikd_tree的降采样参数
                    feats_down_world->resize(feats_down_size);//将下采样得到的地图点大小与body系大小一致
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));//将下采样得到的地图点转换为世界系下的点云
                    }
                    ikdtree_ptr->Build(feats_down_world->points);//从零构建ikd树
                    sig_ikdtreeptr.notify_all();
                }
                continue;
            }
            int featsFromMapNum = ikdtree_ptr->validnum();// 获取ikd tree中的有效节点数，无效点就是被打了deleted标签的点
            kdtree_size_st = ikdtree_ptr->size();//获取ikd tree的节点数
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);
            
            V3D ext_euler = SO3ToEuler(kf_state.offset_R_L_I);//外参，旋转矩阵转欧拉角
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< kf_state.pos.transpose()<<" "<<ext_euler.transpose() << " "<<kf_state.offset_T_L_I.transpose()<< " " << kf_state.vel.transpose() \
            <<" "<<kf_state.bg.transpose()<<" "<<kf_state.ba.transpose()<<" "<<kf_state.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree_ptr->PCL_Storage);//释放PCL_Storage的内存
                ikdtree_ptr->flatten(ikdtree_ptr->Root_Node, ikdtree_ptr->PCL_Storage, NOT_RECORD);//把树展平用于展示
                featsFromMap->clear();
                featsFromMap->points = ikdtree_ptr->PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);//搜索索引
            Nearest_Points.resize(feats_down_size);//将降采样处理后的点云用于搜索最近点
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            {
                unique_lock<std::mutex> my_unique_lock(mtx_ikdtreeptr);
                sig_ikdtreeptr.wait(my_unique_lock, []{return !holding_for_ikdtree_rebuild;});
                kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);//迭代卡尔曼滤波更新，更新地图信息
                sig_ikdtreeptr.notify_all();
            }
            kf_state = kf.get_x();//更新校正之后的系统状态
            euler_cur = SO3ToEuler(kf_state.rot);
            pos_lid = kf_state.pos + kf_state.rot * kf_state.offset_T_L_I;

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();//利用更新后的状态增量式加入kdtree
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
    
            publish_effect_world(pubLaserCloudEffect);
            publish_map(pubLaserCloudMap);

            /* 
                关键帧筛选，子图构建
            */
            /* static state_ikfom last_state;
            V3D position_inc;
            position_inc << kf_state.pos(0) - last_state.pos(0), 
                            kf_state.pos(1) - last_state.pos(1), 
                            kf_state.pos(2) - last_state.pos(2);
            double rotation_inc = kf_state.rot.angularDistance(last_state.rot);
            static PointCloudXYZI::Ptr accum_cloud(new PointCloudXYZI());
            static int submap_frame_num = 5;//一个子图包含的关键帧数量
            static int subframe_idx = 0;//子图中的关键帧索引
            static int submap_idx = 0;//子图索引
            static std::deque< std::tuple<int,state_ikfom,PointCloudXYZI::Ptr> > submaps;//子图信息
            if(position_inc.norm() >= position_threshold || rotation_inc >= rotation_threshold){
                last_state = kf_state;
                //积累n个关键帧作为一个子图
                if(subframe_idx < submap_frame_num){
                    (*accum_cloud) += (*feats_down_world);//累积点云
                }
                else{
                    submap_idx++;//子图索引
                    submaps.push_back(std::tuple(submap_idx, kf_state, accum_cloud));
                    subframe_idx = 0;
                    accum_cloud->clear();
                }
            } */

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree_ptr->size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(kf_state.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_state.pos.transpose()<< " " << ext_euler.transpose() << " "<<kf_state.offset_T_L_I.transpose()<<" "<< kf_state.vel.transpose() \
                <<" "<<kf_state.bg.transpose()<<" "<<kf_state.ba.transpose()<<" "<<kf_state.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }

        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
