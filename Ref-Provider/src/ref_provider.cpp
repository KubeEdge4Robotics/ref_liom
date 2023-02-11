#include "common.h"
// PCL 的相关的头文件
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/io.h>
#include <pcl/visualization/cloud_viewer.h>  
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/kdtree/kdtree_flann.h>
//滤波的头文件
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h> //分割bounding box头文件
#include <pcl/filters/statistical_outlier_removal.h> //统计滤波 头文件
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>

using namespace std;

ros::Publisher pub_prior_constraint;
ros::Publisher pub_prior_lidar_pts;
ros::Publisher pub_prior_mesh_pts_aligned;
ros::Publisher pub_prior_mesh_pts_origin;
nav_msgs::Path  path;
ros::Publisher  ground_truthpub;
std::mutex mtx_models;//模型容器互斥量
double bbox_scale;//模型bbox的缩放因子
int segment_num_threshold;//分割得到的模型点云的最少数目
int opti_num = 2;//ICP迭代次数
//先验模型
std::vector< std::shared_ptr<PriorModel> > prior_models;//先验模型容器
pcl::PointCloud<pcl::PointXYZ> model_pos_cloud;//模型位置点云
//Construct kd-tree object 
pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_model_pos(new pcl::KdTreeFLANN<pcl::PointXYZ>());//模型位置点云kdtree

std::shared_ptr<Mesh> trucky(new Mesh);
std::shared_ptr< igl::AABB<Eigen::MatrixXd,3> > trucky_tree_ptr(new igl::AABB<Eigen::MatrixXd,3>);

std::shared_ptr<PriorModel> modelRecognition(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented){
  std::shared_ptr<PriorModel> prior_model(new PriorModel);
  prior_model->mesh_ptr = trucky;
  prior_model->tree_ptr = trucky_tree_ptr;
  Eigen::Vector4f min_pt(-10, -10.5, -3, 1);
  Eigen::Vector4f max_pt(0, 10.5, 10, 1);
  pcl::CropBox<pcl::PointXYZ> crop_filter;
  crop_filter.setInputCloud(cloud_in);
  crop_filter.setMin(min_pt);
  crop_filter.setMax(max_pt);
  crop_filter.filter(*cloud_segmented);
  return prior_model;
}

void estimateOBB(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::shared_ptr<PriorModel> prior_model){
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud_in, centroid);
  //去中心
  for(auto point : cloud_in->points){
    point.x -= centroid(0);
    point.y -= centroid(1);
    point.z -= centroid(2);
  }
  Point2meshICP point2mesh_icp;
  point2mesh_icp(cloud_in, prior_model);
  Eigen::Affine3d T_w2m = point2mesh_icp.transform.translate(centroid.head<3>().cast<double>());
  double x,y,z,roll,pitch,yaw;
  pcl::getTranslationAndEulerAngles(T_w2m, x, y, z, roll, pitch, yaw);
  prior_model->bbox.trans << x, y, z;
  prior_model->bbox.rot << roll, pitch, yaw;
  // 估计bbox的尺寸
  // 将矩阵转换为pcl点云 
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  // 遍历矩阵
  Eigen::MatrixXd &vertex_matrix = prior_model->mesh_ptr->vertices;
  uint32_t vertex_num = vertex_matrix.rows();
  for (uint32_t i = 0; i < vertex_num; i++) {
  // 将点添加到点云中 
      cloud->points.push_back(pcl::PointXYZ( vertex_matrix(i,0), vertex_matrix(i,1), vertex_matrix(i,2) ));
  }
  pcl::PointXYZ min_pt, max_pt;
	pcl::getMinMax3D(*cloud, min_pt, max_pt);
  prior_model->bbox.min_pt << min_pt.x, min_pt.y, min_pt.z;
  prior_model->bbox.max_pt << max_pt.x, max_pt.y, max_pt.z;
}

void ground_truthcallback(const nav_msgs::Odometry::ConstPtr& ground_truth)
{
    geometry_msgs::PoseStamped this_pose_stamped;
    this_pose_stamped.pose.position.x = ground_truth->pose.pose.position.x;
    this_pose_stamped.pose.position.y = ground_truth->pose.pose.position.y;
    this_pose_stamped.pose.orientation = ground_truth->pose.pose.orientation;
    this_pose_stamped.header.stamp = ros::Time::now();
    this_pose_stamped.header.frame_id = "camera_init";

    path.poses.push_back(this_pose_stamped);
 
    path.header.stamp = ros::Time::now();
    path.header.frame_id="camera_init";
    ground_truthpub.publish(path);
}

void priorCloudHandler (const nav_msgs::Odometry::ConstPtr &msg_odom, const sensor_msgs::PointCloud2ConstPtr& msg_cloud)  
{ 
  ROS_INFO("\033[1;31m xxxxPriorCloudHandlerStart, %f  \033[0m", ros::Time::now().toSec());
  // PCL 
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_body(new pcl::PointCloud<pcl::PointXYZ>); //pcl_cloud_msg要转化成的点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_world(new pcl::PointCloud<pcl::PointXYZ>); 
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_model(new pcl::PointCloud<pcl::PointXYZ>); 
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented_body (new pcl::PointCloud<pcl::PointXYZ>);//分割算法得到的点云(body系)
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented_world (new pcl::PointCloud<pcl::PointXYZ>);//分割得到的点云(世界系)
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented_model (new pcl::PointCloud<pcl::PointXYZ>);//分割得到的点云(model系)
  double var = 0;//协方差
  uint32_t keypose_idx = msg_cloud->header.seq;//关键帧id
	// 转化：ROS 点云 -> PCL 
	pcl::fromROSMsg(*msg_cloud, *pcl_cloud_body);
  // 坐标转换body_to_world
  Eigen::Affine3d T_b2w = poseMsg2Affine3d(msg_odom->pose.pose);
  pcl::transformPointCloud(*pcl_cloud_body, *pcl_cloud_world, T_b2w.matrix());
  pcl_cloud_body.reset();
  //Define the circular range for search
  float radius = 0.01f;
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  pcl::PointXYZ query_pos;//查询位置
  query_pos.x = msg_odom->pose.pose.position.x;
  query_pos.y = msg_odom->pose.pose.position.y;
  query_pos.z = msg_odom->pose.pose.position.z;
  //Search the kd-tree
  unsigned int results = kdtree_model_pos->radiusSearch(query_pos, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
  //对于每一个可能提取到的先验模型
  for(size_t i = 0;i < results; i++){
    int model_idx = pointIdxRadiusSearch[i];//模型的索引
    std::shared_ptr<PriorModel> &prior_model = prior_models[model_idx];
    Eigen::Affine3d T_w2m;//世界系->模型系
    {
      std::unique_lock<std::mutex> ulock(mtx_models);
      pcl::getTransformation(prior_model->bbox.trans(0), prior_model->bbox.trans(1), prior_model->bbox.trans(2),\
                            prior_model->bbox.rot(0), prior_model->bbox.rot(1), prior_model->bbox.rot(2), T_w2m);
    }
    pcl::transformPointCloud(*pcl_cloud_world, *pcl_cloud_model, T_w2m);// 分割点云由世界系转换到模型系下
    pcl::IndicesPtr segmented_indices(new std::vector<int>);//分割点云的索引
    //设置bounding box的角点
    Eigen::Vector4f min_pt(bbox_scale * prior_model->bbox.min_pt(0), bbox_scale * prior_model->bbox.min_pt(1), bbox_scale * prior_model->bbox.min_pt(2), 1);
    Eigen::Vector4f max_pt(bbox_scale * prior_model->bbox.max_pt(0), bbox_scale * prior_model->bbox.max_pt(1), bbox_scale * prior_model->bbox.max_pt(2), 1);
    // Create the crop box filter and apply it
    pcl::CropBox<pcl::PointXYZ> crop_filter;
    crop_filter.setInputCloud(pcl_cloud_model);
    crop_filter.setMin(min_pt);
    crop_filter.setMax(max_pt);
    crop_filter.filter(*segmented_indices);
    //如果分割点云过少，则跳过当前循环
    if(segmented_indices->size() < segment_num_threshold) continue;
    // 分割后点云(model系)
    pcl::copyPointCloud(*pcl_cloud_model, *segmented_indices, *cloud_segmented_model);
    // 分割后点云(world系)
    pcl::copyPointCloud(*pcl_cloud_world, *segmented_indices, *cloud_segmented_world);
    //提取剩余点云
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(pcl_cloud_world);
    extract.setIndices(segmented_indices);
    extract.setNegative(true);//设置为排除索引提供的点云
    extract.filter(*pcl_cloud_world);
    // 搜索虚拟点+点云网格配准 
    Eigen::VectorXd sqrD;//list of squared distances
    Eigen::VectorXi indices_facets; //list of indices into Element of closest mesh primitive
    Eigen::MatrixXd closest_points;//list of closest points
    // ceres ICP
    for(int i=1;i<=opti_num;i++){      
      Eigen::MatrixXd query_points(cloud_segmented_model->points.size(), 3);//query_points : list of query points
      for (int i=0; i  < cloud_segmented_model->points.size(); i++) {
        query_points(i, 0) = cloud_segmented_model->points[i].x;
        query_points(i, 1) = cloud_segmented_model->points[i].y;
        query_points(i, 2) = cloud_segmented_model->points[i].z;
      }
      // 更新mesh匹配
      pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      TicToc t_point_mesh;//搜索匹配点的时间
      prior_model->tree_ptr->squared_distance(prior_model->mesh_ptr->vertices, prior_model->mesh_ptr->facets, query_points, sqrD, indices_facets, closest_points);
      std::cout << "prior_lidar_pts size: " << cloud_segmented_model->points.size()<< "\n"
                << "point-mesh time (ms): " << t_point_mesh.toc() <<std::endl;
      int meshnum = closest_points.size()/3;//mesh虚拟点数目
      for (int i=0; i< meshnum; i++) {
        pcl::PointXYZ pointc;//匹配最近点
        pointc.x = closest_points(i,0);
        pointc.y = closest_points(i,1);
        pointc.z = closest_points(i,2);
        mesh_cloud->points.push_back(pointc);
      }
      double scale_mesh = 1.0/meshnum;//point-to-mesh权重系数
      // 构造损失函数
      double parameter[7] = {0, 0, 0, 1, 0, 0, 0};//迭代初值
      Eigen::Map<Eigen::Quaterniond> q(parameter);
      Eigen::Map<Eigen::Vector3d> t(parameter + 4);
      ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
      ceres::LossFunction *loss_function_scaled = 
        new ceres::ScaledLoss(loss_function, scale_mesh, ceres::Ownership::DO_NOT_TAKE_OWNERSHIP);
      ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
      ceres::Problem::Options problem_options;
      ceres::Problem problem(problem_options);
      problem.AddParameterBlock(parameter, 4, q_parameterization);	// 添加参数块，旋转四元数
      problem.AddParameterBlock(parameter + 4, 3);
      for (int i = 0; i < meshnum; i++)
      {
        pcl::PointXYZ &lidarPt = cloud_segmented_model->points[i];
        pcl::PointXYZ &meshPt = mesh_cloud->points[i];
        Eigen::Vector3d lidar_pt(lidarPt.x, lidarPt.y, lidarPt.z);
        Eigen::Vector3d mesh_pt(meshPt.x, meshPt.y, meshPt.z);
        ceres::CostFunction *cost_function = PointMeshFactor::Create(lidar_pt, mesh_pt);
        problem.AddResidualBlock(cost_function, loss_function_scaled, parameter, parameter + 4);
      }

      TicToc t_solver;
      ceres::Solver::Options options;
      options.minimizer_type = ceres::TRUST_REGION;
      options.linear_solver_type = ceres::DENSE_QR;
      options.max_num_iterations = 4;
      options.minimizer_progress_to_stdout = false;
      options.check_gradients = false;
      options.gradient_check_relative_precision = 1e-4;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      var = summary.final_cost;//最终函数值
      // std::cout << summary.BriefReport() << "\n";

      //model在世界系下的位姿
      Eigen::Affine3d T_inc=Eigen::Affine3d::Identity();//位姿增量
      T_inc.rotate(q);
      T_inc.pretranslate(t);
      T_w2m = T_inc * T_w2m;//更新T_w2m
      // 更新原始点云world_to_model
      pcl::transformPointCloud(*cloud_segmented_world, *cloud_segmented_model, T_w2m);// 更新之后的模型系点云
      //收缩bounding box
      min_pt(0) = bbox_scale * prior_model->bbox.min_pt(0);
      min_pt(1) = bbox_scale * prior_model->bbox.min_pt(1);
      min_pt(2) = bbox_scale * prior_model->bbox.min_pt(2);
      max_pt(0) = bbox_scale * prior_model->bbox.max_pt(0);
      max_pt(1) = bbox_scale * prior_model->bbox.max_pt(1);
      max_pt(2) = bbox_scale * prior_model->bbox.max_pt(2);
      // 重新分割模型点云
      pcl::CropBox<pcl::PointXYZ> crop_filter;
      crop_filter.setInputCloud(cloud_segmented_model);
      crop_filter.setMin(min_pt);
      crop_filter.setMax(max_pt);
      crop_filter.filter(*segmented_indices);
      // 将分割剩余点云放回pcl_cloud_world供下一个模型提取
      pcl::PointCloud<pcl::PointXYZ> cloud_add;
      extract.setInputCloud(cloud_segmented_world);
      extract.setIndices(segmented_indices);
      extract.setNegative(true);//设置为排除索引提供的点云
      extract.filter(cloud_add);
      *pcl_cloud_world += cloud_add;//添加分割剩余点云
      // 提取分割后点云(model系)
      extract.setInputCloud(cloud_segmented_model);
      extract.setIndices(segmented_indices);
      extract.setNegative(false);//设置为提取索引提供的点云
      extract.filter(*cloud_segmented_model);
      //提取分割后点云(world系)
      extract.setInputCloud(cloud_segmented_world);
      extract.setIndices(segmented_indices);
      extract.setNegative(false);//设置为提取索引提供的点云
      extract.filter(*cloud_segmented_world);
      
      //ICP迭代结束，点云发布
      std::cout<<"ICP update finish"<<std::endl;
      // 分割后的点云（世界坐标系）、迭代前的虚拟点云、迭代后的虚拟点云
      publishCloud(pub_prior_lidar_pts, cloud_segmented_world, ros::Time::now(), "camera_init");
      // publishCloud(pub_prior_mesh_pts_origin, mesh_cloud, ros::Time::now(), "camera_init");
      // 迭代后的虚拟点云转换会world系发布
      pcl::transformPointCloud(*mesh_cloud, *mesh_cloud, T_w2m.inverse());// 更新之后的模型系点云
      publishCloud(pub_prior_mesh_pts_aligned, mesh_cloud, ros::Time::now(), "camera_init");
    }
    //发送模型与雷达帧的约束
    geometry_msgs::PoseWithCovariance msg_constraint;
    Eigen::Affine3d T_m2b = T_b2w * T_w2m.inverse();
    msg_constraint.pose = Affine3d2poseMsg(T_m2b); 
    msg_constraint.covariance[0] = model_idx;//模型id
    msg_constraint.covariance[1] = keypose_idx;//关键帧id
    msg_constraint.covariance[2] = 0.1 * var;//位移协方差
    msg_constraint.covariance[3] = var;//旋转协方差
    pub_prior_constraint.publish(msg_constraint);
  }

  //对于剩下的点云进行分割，得到在world系下模型扫描点云模型 
  std::shared_ptr<PriorModel> prior_model = modelRecognition(pcl_cloud_world, cloud_segmented_world);
  //将模型放入容器
  prior_models.push_back(prior_model);
  estimateOBB(cloud_segmented_world, prior_model);
}



void modelPosesCallback(const nav_msgs::Path::ConstPtr &msg){
  //校正模型位姿
  std::unique_lock<std::mutex> ulock(mtx_models);
  for(auto &pose : msg->poses){
    float x,y,z,roll,pitch,yaw;
    pcl::getTranslationAndEulerAngles(poseMsg2Affine3f(pose.pose), x, y, z, roll, pitch, yaw);
    prior_models[pose.header.seq]->bbox.trans = Eigen::Vector3f(x, y, z);
    prior_models[pose.header.seq]->bbox.rot = Eigen::Vector3f(roll, pitch, yaw);
  }
}

int main (int argc, char** argv)
{
  igl::readOBJ("/home/long/Code/RLOAM/models/trucky-10.obj", trucky->vertices, trucky->facets); 
  trucky_tree_ptr->init(trucky->vertices, trucky->facets);
  ROS_INFO("\033[1;31m **********Point-Mesh Thread Started********** \033[0m");

  ros::init (argc, argv, "ref_provider");
  ros::NodeHandle nh;
  ros::MultiThreadedSpinner spinner(2);
  nh.param<double>("ref_provider/bbox_scale", bbox_scale, 1);
  nh.param<int>("ref_provider/segment_num_threshold", segment_num_threshold, 500);
  nh.param<int>("ref_provider/mapping_opti_num", opti_num, 2);
  std::shared_ptr< message_filters::Subscriber<nav_msgs::Odometry> > subOdom;
  std::shared_ptr< message_filters::Subscriber<sensor_msgs::PointCloud2> > subCloud;
  typedef message_filters::sync_policies::ExactTime<nav_msgs::Odometry, sensor_msgs::PointCloud2> SyncPolicy;
  std::shared_ptr< message_filters::Synchronizer<SyncPolicy> > odom_cloud_sync;
  subOdom = std::make_shared< message_filters::Subscriber<nav_msgs::Odometry> >(nh, "aftPgo_Odom", 10);
  subCloud = std::make_shared< message_filters::Subscriber<sensor_msgs::PointCloud2> >(nh, "submap", 10);
  odom_cloud_sync = std::make_shared< message_filters::Synchronizer<SyncPolicy> >(SyncPolicy(10), *subOdom, *subCloud);//10是消息队列长度
  odom_cloud_sync->registerCallback(boost::bind(&priorCloudHandler, _1, _2));
  ros::Subscriber subModelPoses = nh.subscribe<nav_msgs::Path>("model_poses", 10, modelPosesCallback);//先验模型优化位姿
  ros::Subscriber subgroundtruth = nh.subscribe<nav_msgs::Odometry> ("base_pose_ground_truth", 100, ground_truthcallback);

  pub_prior_constraint = nh.advertise<geometry_msgs::PoseWithCovariance> ("prior_constraints", 10);//先验模型约束
  pub_prior_lidar_pts = nh.advertise<sensor_msgs::PointCloud2> ("prior_lidar_pts", 10);
  pub_prior_mesh_pts_aligned = nh.advertise<sensor_msgs::PointCloud2>("prior_mesh_pts_aligned",10);
  pub_prior_mesh_pts_origin = nh.advertise<sensor_msgs::PointCloud2>("prior_mesh_pts_origin",10);
  ground_truthpub = nh.advertise<nav_msgs::Path>("ground_truth",10, true);

  spinner.spin();
  return 0;
}
