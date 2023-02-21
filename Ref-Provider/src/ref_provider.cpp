#include "common.h"
#include "pcl_utils.hpp"
#include <ros/package.h>
#include <json/json.h>
#include <igl/read_triangle_mesh.h>
#include <igl/per_face_normals.h>
#include <igl/random_points_on_mesh.h>
using namespace std;

ros::Publisher pub_prior_constraint;
ros::Publisher pub_prior_lidar_pts;
ros::Publisher pub_prior_mesh_pts_aligned;
ros::Publisher pub_prior_mesh_pts_origin;
ros::Publisher pub_prior_constraint_edge;
ros::Publisher pub_model_cloud_normal;
ros::Publisher pub_ground_cloud;

nav_msgs::Path path;
ros::Publisher ground_truthpub;
ros::Publisher pub_model_pose;
std::mutex mtx_models;					// 模型容器互斥量
double track_bbox_scale;				// 模型bbox的缩放因子
int segment_num_threshold;			// 分割得到的模型点云的最少数目
int max_opti_num;								// ICP迭代次数
float search_radius;						// Define the circular range for search
double fitness_threshold;				// point-mesh-icp评估阈值
double max_correspondence_dist; // point-mesh匹配最大距离
string map_frame;
string point_cloud_topic;
string aftPgo_odom_topic;
string model_poses_topic;

// 先验模型
std::vector<std::shared_ptr<PriorModel>> prior_models;																	 // 先验模型容器
pcl::PointCloud<pcl::PointXYZ>::Ptr model_pos_cloud(new pcl::PointCloud<pcl::PointXYZ>); // 模型位置点云
// Construct kd-tree object
pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_model_pos(new pcl::KdTreeFLANN<pcl::PointXYZ>()); // 模型位置点云kdtree

std::shared_ptr<PriorModel> modelRecognition(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented)
{
	std::shared_ptr<PriorModel> prior_model(new PriorModel);
	Eigen::Vector4f min_pt(-10, -10.5, -3, 1);
	Eigen::Vector4f max_pt(0, 10.5, 10, 1);
	pcl::CropBox<pcl::PointXYZ> crop_filter;
	crop_filter.setInputCloud(cloud_in);
	crop_filter.setMin(min_pt);
	crop_filter.setMax(max_pt);
	crop_filter.filter(*cloud_segmented);
	return prior_model;
}

void estimateOBB(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::shared_ptr<PriorModel> prior_model)
{
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*cloud_in, centroid);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented_model(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_cloud(new pcl::PointCloud<pcl::PointXYZ>); // mesh虚拟匹配点
	// 世界系 -> model自身坐标系
	Eigen::Affine3d T_w2m = Eigen::Affine3d::Identity();
	T_w2m.pretranslate(Eigen::Vector3d(-centroid(0), -centroid(1), -centroid(2)));
	Point2meshICP point2mesh_icp;
	for (int i = 1; i <= max_opti_num; i++)
	{
		// 更新原始点云world_to_model
		pcl::transformPointCloud(*cloud_in, *cloud_segmented_model, T_w2m); // 更新之后的模型系点云
		point2mesh_icp(cloud_segmented_model, prior_model, mesh_cloud);
		Eigen::Affine3d T_inc = point2mesh_icp.getTransform(); // 位姿增量
		T_w2m = T_inc * T_w2m;																 // 更新T_w2m
	}
	double x, y, z, roll, pitch, yaw;
	pcl::getTranslationAndEulerAngles(T_w2m, x, y, z, roll, pitch, yaw);
	prior_model->bbox.trans << x, y, z;
	prior_model->bbox.rot << roll, pitch, yaw;
	// 将模型位置加入model_pos_cloud中
	model_pos_cloud->push_back(pcl::PointXYZ(x, y, z));
	// 估计bbox的尺寸
	// 将矩阵转换为pcl点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	// 遍历矩阵
	Eigen::MatrixXd &vertex_matrix = prior_model->mesh_ptr->vertices;
	uint32_t vertex_num = vertex_matrix.rows();
	for (uint32_t i = 0; i < vertex_num; i++)
	{
		// 将点添加到点云中
		cloud->points.push_back(pcl::PointXYZ(vertex_matrix(i, 0), vertex_matrix(i, 1), vertex_matrix(i, 2)));
	}
	pcl::PointXYZ min_pt, max_pt;
	pcl::getMinMax3D(*cloud, min_pt, max_pt);
	prior_model->bbox.min_pt << min_pt.x, min_pt.y, min_pt.z;
	prior_model->bbox.max_pt << max_pt.x, max_pt.y, max_pt.z;
}

void ground_truthcallback(const nav_msgs::Odometry::ConstPtr &ground_truth)
{
	geometry_msgs::PoseStamped this_pose_stamped;
	this_pose_stamped.pose.position.x = ground_truth->pose.pose.position.x;
	this_pose_stamped.pose.position.y = ground_truth->pose.pose.position.y;
	this_pose_stamped.pose.orientation = ground_truth->pose.pose.orientation;
	this_pose_stamped.header.stamp = ros::Time::now();
	this_pose_stamped.header.frame_id = map_frame;

	path.poses.push_back(this_pose_stamped);

	path.header.stamp = ros::Time::now();
	path.header.frame_id = map_frame;
	ground_truthpub.publish(path);
}
void visualizePriorConstraint(const ros::Time &time_stamp, const Eigen::Vector3d &trans_b2w, const std::vector<Eigen::Vector3d> &model_pos)
{
	visualization_msgs::MarkerArray markerArray;
	// loop nodes
	visualization_msgs::Marker markerNode;
	markerNode.header.frame_id = map_frame;
	markerNode.header.stamp = time_stamp;
	markerNode.action = visualization_msgs::Marker::ADD;
	markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
	markerNode.ns = "prior_nodes";
	markerNode.id = 0;
	markerNode.pose.orientation.w = 1;
	markerNode.scale.x = 0.3;
	markerNode.scale.y = 0.3;
	markerNode.scale.z = 0.3;
	markerNode.color.r = 0;
	markerNode.color.g = 0;
	markerNode.color.b = 1;
	markerNode.color.a = 1;
	// loop edges
	visualization_msgs::Marker markerEdge;
	markerEdge.header.frame_id = map_frame;
	markerEdge.header.stamp = time_stamp;
	markerEdge.action = visualization_msgs::Marker::ADD;
	markerEdge.type = visualization_msgs::Marker::LINE_LIST;
	markerEdge.ns = "prior_edges";
	markerEdge.id = 1;
	markerEdge.pose.orientation.w = 1;
	markerEdge.scale.x = 0.1;
	markerEdge.color.r = 0;
	markerEdge.color.g = 0.9;
	markerEdge.color.b = 0;
	markerEdge.color.a = 1;

	for (auto &trans_m2w : model_pos)
	{
		geometry_msgs::Point p;
		p.x = trans_b2w.x();
		p.y = trans_b2w.y();
		p.z = trans_b2w.z();
		markerNode.points.push_back(p);
		markerEdge.points.push_back(p);
		p.x = trans_m2w.x();
		p.y = trans_m2w.y();
		p.z = trans_m2w.z();
		markerNode.points.push_back(p);
		markerEdge.points.push_back(p);
		// 模型绝对位姿
		p.x = -prior_models.front()->bbox.trans.x();
		p.y = -prior_models.front()->bbox.trans.y();
		p.z = -prior_models.front()->bbox.trans.z();
		markerNode.points.push_back(p);
	}

	markerArray.markers.push_back(markerNode);
	markerArray.markers.push_back(markerEdge);
	pub_prior_constraint_edge.publish(markerArray);
}
void syncedCallback(const nav_msgs::Odometry::ConstPtr &msg_odom, const sensor_msgs::PointCloud2ConstPtr &msg_cloud)
{
	ROS_INFO("\033[1;31m xxxxsyncedCallbackStart, %f  \033[0m", ros::Time::now().toSec());
	// 可视化用的
	std::vector<Eigen::Vector3d> model_pos;
	// PCL
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_body(new pcl::PointCloud<pcl::PointXYZ>); // pcl_cloud_msg要转化成的点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_world(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_model(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented_body(new pcl::PointCloud<pcl::PointXYZ>);	 // 分割算法得到的点云(body系)
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented_world(new pcl::PointCloud<pcl::PointXYZ>); // 分割得到的点云(世界系)
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented_model(new pcl::PointCloud<pcl::PointXYZ>); // 分割得到的点云(model系)
	uint32_t keypose_idx = msg_cloud->header.seq;																									 // 关键帧id
	// 转化：ROS 点云 -> PCL
	pcl::fromROSMsg(*msg_cloud, *pcl_cloud_body);
	pcl::IndicesPtr indices_plane(new pcl::Indices);
	// 分割剔除地面点
	segmentPlane(pcl_cloud_body, *indices_plane);
	pcl::PointCloud<pcl::PointXYZ>::Ptr ground_points(new pcl::PointCloud<pcl::PointXYZ>);
	// 提取非平面点云
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	//提取平面点
	extract.setInputCloud(pcl_cloud_body);
	extract.setIndices(indices_plane);
	extract.setNegative(false); // 设置为剔除索引提供的点云
	extract.filter(*ground_points);
	//提取非平面点
	extract.setInputCloud(pcl_cloud_body);
	extract.setIndices(indices_plane);
	extract.setNegative(true); // 设置为剔除索引提供的点云
	extract.filter(*pcl_cloud_body);
	// 发布地面点云
	publishCloud(pub_ground_cloud, ground_points, ros::Time::now(), map_frame);
	// 坐标转换body_to_world
	Eigen::Affine3d T_b2w = poseMsg2Affine3d(msg_odom->pose.pose);
	pcl::transformPointCloud(*pcl_cloud_body, *pcl_cloud_world, T_b2w.matrix());
	pcl_cloud_body.reset();

	// 估计法向量
	pcl::PointCloud<pcl::Normal>::Ptr cloud_world_normals = computeNormals<pcl::PointXYZ>(pcl_cloud_world, 2.0);

	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;
	pcl::PointXYZ query_pos; // 查询位置
	query_pos.x = msg_odom->pose.pose.position.x;
	query_pos.y = msg_odom->pose.pose.position.y;
	query_pos.z = msg_odom->pose.pose.position.z;
	// Search the kd-tree
	kdtree_model_pos->setInputCloud(model_pos_cloud);
	unsigned int results = kdtree_model_pos->radiusSearch(query_pos, search_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
	// 对于每一个可能提取到的先验模型
	for (size_t i = 0; i < results; i++)
	{
		int model_idx = pointIdxRadiusSearch[i]; // 模型的索引
		std::shared_ptr<PriorModel> &prior_model = prior_models[model_idx];
		Eigen::Affine3d T_w2m; // 世界系->模型系
		{
			std::unique_lock<std::mutex> ulock(mtx_models);
			pcl::getTransformation(prior_model->bbox.trans(0), prior_model->bbox.trans(1), prior_model->bbox.trans(2),
														 prior_model->bbox.rot(0), prior_model->bbox.rot(1), prior_model->bbox.rot(2), T_w2m);
		}
		pcl::transformPointCloud(*pcl_cloud_world, *pcl_cloud_model, T_w2m); // 分割点云由世界系转换到模型系下
		pcl::IndicesPtr segmented_indices(new std::vector<int>);						 // 分割点云的索引
		// 设置跟踪bounding box
		Eigen::Vector4f min_pt = Eigen::Vector4f::Ones();
		Eigen::Vector4f max_pt = Eigen::Vector4f::Ones();
		min_pt.head<3>() = track_bbox_scale * prior_model->bbox.min_pt;
		max_pt.head<3>() = track_bbox_scale * prior_model->bbox.max_pt;
		// Create the crop box filter and apply it
		pcl::CropBox<pcl::PointXYZ> crop_filter;
		crop_filter.setInputCloud(pcl_cloud_model);
		crop_filter.setMin(min_pt);
		crop_filter.setMax(max_pt);
		crop_filter.filter(*segmented_indices);
		// 如果分割点云过少，则跳过当前循环
		if (segmented_indices->size() < segment_num_threshold)
		{
			cout << "too few segmented points: " << segmented_indices->size() << endl;
			continue;
		}
		// 分割后点云(world系)
		pcl::copyPointCloud(*pcl_cloud_world, *segmented_indices, *cloud_segmented_world);
		// 提取剩余点云
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(pcl_cloud_world);
		extract.setIndices(segmented_indices);
		extract.setNegative(true); // 设置为排除索引提供的点云
		extract.filter(*pcl_cloud_world);
		double shrink_scale = 1; // bbox收缩因子
		double final_scale = 0.2;
		double k = log(final_scale) / static_cast<double>(max_opti_num);
		Point2meshICP point2mesh_icp; // 点到mesh ICP
		// point2mesh_icp.setRotCostant(); // 设置旋转不变
		// 设置最大匹配距离random_points_on_mesh
		point2mesh_icp.setMaxCorrespondenceDist(max_correspondence_dist);
		// 设置异常值阈值
		point2mesh_icp.setOutlierThreshold(0.2);
		pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_cloud(new pcl::PointCloud<pcl::PointXYZ>); // mesh匹配虚拟点
		for (int i = 0; i < max_opti_num; i++)
		{
			// 更新原始点云world_to_model
			pcl::transformPointCloud(*cloud_segmented_world, *cloud_segmented_model, T_w2m); // 更新之后的模型系点云
			// 收缩bounding box
			min_pt.head<3>() = shrink_scale * track_bbox_scale * prior_model->bbox.min_pt;
			max_pt.head<3>() = shrink_scale * track_bbox_scale * prior_model->bbox.max_pt;
			// 重新分割模型点云
			pcl::CropBox<pcl::PointXYZ> crop_filter;
			crop_filter.setInputCloud(cloud_segmented_model);
			crop_filter.setMin(min_pt);
			crop_filter.setMax(max_pt);
			crop_filter.filter(*segmented_indices);
			// 如果分割点云过少，则跳过当前循环
			if (segmented_indices->size() < segment_num_threshold)
			{
				cout << "too few segmented points: " << segmented_indices->size() << endl;
				break;
			}
			// 将分割剩余点云放回pcl_cloud_world供下一个模型提取
			pcl::PointCloud<pcl::PointXYZ> cloud_add;
			extract.setInputCloud(cloud_segmented_world);
			extract.setIndices(segmented_indices);
			extract.setNegative(true); // 设置为排除索引提供的点云
			extract.filter(cloud_add);
			*pcl_cloud_world += cloud_add; // 添加分割剩余点云
			// 提取分割后点云(model系)
			extract.setInputCloud(cloud_segmented_model);
			extract.setIndices(segmented_indices);
			extract.setNegative(false); // 设置为提取索引提供的点云
			extract.filter(*cloud_segmented_model);
			// 提取分割后点云(world系)
			extract.setInputCloud(cloud_segmented_world);
			extract.setIndices(segmented_indices);
			extract.setNegative(false); // 设置为提取索引提供的点云
			extract.filter(*cloud_segmented_world);

			// point-mesh icp
			if (!point2mesh_icp(cloud_segmented_model, prior_model, mesh_cloud))
			{
				// 如果异常点过多，调整缩放因子，再次分割
				cout << "outlier too much" << endl;
				// shrink_scale *= 0.9;
				continue;
			}

			// 只有当收敛了才会更新
			if (point2mesh_icp.hasConverged() && point2mesh_icp.getFitness() < fitness_threshold)
			{
				// 设置最大匹配距离
				point2mesh_icp.setMaxCorrespondenceDist(max_correspondence_dist * exp(k * static_cast<double>(i + 1)));
				Eigen::Affine3d T_inc = point2mesh_icp.getTransform(); // 位姿增量
				T_w2m = T_inc * T_w2m;																 // 更新T_w2m
				// ICP迭代结束，点云发布
				std::cout << "ICP update finish" << std::endl;
			}
			// else
			// {
			// 	// 如果ICP结果不好，调整，重新ICP
			// 	shrink_scale *= 0.9;
			// }
		}
		if (point2mesh_icp.hasConverged() && point2mesh_icp.getFitness() < 0.01 * fitness_threshold)
		{
			// 发送模型与雷达帧的约束
			geometry_msgs::PoseWithCovariance msg_constraint;
			Eigen::Affine3d T_m2b = (T_b2w * T_w2m).inverse(); // model->body约束
			msg_constraint.pose = Affine3d2poseMsg(T_m2b);
			msg_constraint.covariance[0] = keypose_idx;									// 关键帧id
			msg_constraint.covariance[1] = model_idx;										// 模型id
			msg_constraint.covariance[2] = point2mesh_icp.getFitness(); // 位移协方差
			msg_constraint.covariance[3] = point2mesh_icp.getFitness(); // 旋转协方差
			pub_prior_constraint.publish(msg_constraint);
			// 可视化用的
			model_pos.push_back(T_w2m.inverse().translation());
			// 发布模型位姿
			nav_msgs::Odometry::Ptr msg(new nav_msgs::Odometry);
			msg->header.stamp = ros::Time::now();
			msg->header.frame_id = map_frame;
			msg->child_frame_id = "model";
			msg->pose.pose = Affine3d2poseMsg(T_w2m.inverse());
			pub_model_pose.publish(msg);
		}
		else
		{
			// 分割后的点云（世界坐标系）、迭代前的虚拟点云、迭代后的虚拟点云
			publishCloud(pub_prior_lidar_pts, cloud_segmented_world, ros::Time::now(), map_frame);
			// 迭代后的虚拟点云转换会world系发布
			pcl::transformPointCloud(*mesh_cloud, *mesh_cloud, T_w2m.inverse()); // 更新之后的模型系点云
			publishCloud(pub_prior_mesh_pts_aligned, mesh_cloud, ros::Time::now(), map_frame);
		}
	}
	// 可视化先验约束
	if (model_pos.size() > 0)
		visualizePriorConstraint(msg_odom->header.stamp, T_b2w.translation(), model_pos);
	// // 对于剩下的点云进行分割，得到在world系下模型扫描点云模型
	// std::shared_ptr<PriorModel> prior_model = modelRecognition(pcl_cloud_world, cloud_segmented_world);
	// // 将模型放入容器
	// prior_models.push_back(prior_model);
	// estimateOBB(cloud_segmented_world, prior_model);
}

void modelPosesCallback(const nav_msgs::Path::ConstPtr &msg)
{
	// 校正模型位姿
	std::unique_lock<std::mutex> ulock(mtx_models);
	for (auto &pose : msg->poses)
	{
		double x, y, z, roll, pitch, yaw;
		pcl::getTranslationAndEulerAngles(poseMsg2Affine3d(pose.pose).inverse(), x, y, z, roll, pitch, yaw);
		prior_models[pose.header.seq]->bbox.trans << x, y, z;
		prior_models[pose.header.seq]->bbox.rot << roll, pitch, yaw;
		model_pos_cloud->points[pose.header.seq].x = x;
		model_pos_cloud->points[pose.header.seq].y = y;
		model_pos_cloud->points[pose.header.seq].z = z;
	}
}
// 从文件中读取先验信息
bool modelInit(const string &path)
{
	// read json from stream
	std::ifstream ifs;
	ifs.open(path);
	if (!ifs.is_open())
		std::cerr << "ifstream open error!" << std::endl;
	Json::Value root;
	Json::CharReaderBuilder builder;
	builder["collectComments"] = true;
	JSONCPP_STRING errs;
	if (!parseFromStream(builder, ifs, &root, &errs))
	{
		std::cerr << errs << std::endl;
		return false;
	}
	TicToc t_model_init; // 模型初始化的时间
	pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_normal_all(new pcl::PointCloud<pcl::PointXYZINormal>);
	for (auto &m : root["models"])
	{
		std::shared_ptr<Mesh> mesh_ptr(new Mesh);
		igl::read_triangle_mesh(m["path"].asString(), mesh_ptr->vertices, mesh_ptr->facets);
		std::shared_ptr<PriorModel> model_ptr(new PriorModel(mesh_ptr));
		Eigen::Affine3d T_m2w;
		pcl::getTransformation(m["trans"][0].asDouble(), m["trans"][1].asDouble(), m["trans"][2].asDouble(),
													 m["rot"][0].asDouble(), m["rot"][1].asDouble(), m["rot"][2].asDouble(), T_m2w);
		double x, y, z, roll, pitch, yaw;
		pcl::getTranslationAndEulerAngles(T_m2w.inverse(), x, y, z, roll, pitch, yaw);
		model_ptr->bbox.trans << x, y, z;
		model_ptr->bbox.rot << roll, pitch, yaw;
		model_ptr->bbox.min_pt << m["min_pt"][0].asDouble(), m["min_pt"][1].asDouble(), m["min_pt"][2].asDouble();
		model_ptr->bbox.max_pt << m["max_pt"][0].asDouble(), m["max_pt"][1].asDouble(), m["max_pt"][2].asDouble();
		model_pos_cloud->push_back(pcl::PointXYZ(model_ptr->bbox.trans(0), model_ptr->bbox.trans(1), model_ptr->bbox.trans(2)));
		prior_models.push_back(model_ptr);
		// mesh to point cloud with normal vector
		pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_normal(new pcl::PointCloud<pcl::PointXYZINormal>);
		{ // add vertex
			Eigen::MatrixXd N_vertices;
			// Compute per-vertex normals
			igl::per_vertex_normals(mesh_ptr->vertices, mesh_ptr->facets, N_vertices);
			for (int i = 0; i < N_vertices.rows(); i++)
			{
				pcl::PointXYZINormal point_normal;
				point_normal.x = mesh_ptr->vertices(i, 0);
				point_normal.y = mesh_ptr->vertices(i, 1);
				point_normal.z = mesh_ptr->vertices(i, 2);
				point_normal.normal_x = N_vertices(i, 0);
				point_normal.normal_y = N_vertices(i, 1);
				point_normal.normal_z = N_vertices(i, 2);
				cloud_normal->push_back(point_normal);
			}
		}
		{ // samping on facets
			Eigen::MatrixXd N_faces;
			// Compute per-face normals
			igl::per_face_normals(mesh_ptr->vertices, mesh_ptr->facets, N_faces);
			Eigen::MatrixXd barycentric_coord;
			Eigen::VectorXi F_index;
			igl::random_points_on_mesh(5000, mesh_ptr->vertices, mesh_ptr->facets, barycentric_coord, F_index);
			for (int i = 0; i < F_index.size(); i++)
			{
				Eigen::Vector3i vertex_indices = mesh_ptr->facets.row(F_index(i));	 // 面片点索引
				Eigen::Vector3d vertex0 = mesh_ptr->vertices.row(vertex_indices(0)); // 顶点1
				Eigen::Vector3d vertex1 = mesh_ptr->vertices.row(vertex_indices(1)); // 顶点2
				Eigen::Vector3d vertex2 = mesh_ptr->vertices.row(vertex_indices(2)); // 顶点3
				// 采样点
				Eigen::Vector3d vertex = barycentric_coord(i, 0) * vertex0 +
																 barycentric_coord(i, 1) * vertex1 +
																 barycentric_coord(i, 2) * vertex2;
				// 法向量
				Eigen::Vector3d normal = N_faces.row(F_index(i));

				pcl::PointXYZINormal point_normal;
				point_normal.x = vertex(0);
				point_normal.y = vertex(1);
				point_normal.z = vertex(2);
				point_normal.normal_x = normal(0);
				point_normal.normal_y = normal(1);
				point_normal.normal_z = normal(2);
				cloud_normal->push_back(point_normal);
			}
		}
		/* for (int i = 0; i < mesh_ptr->facets.rows(); i++)
		{
			Eigen::Vector3d vertex0 = mesh_ptr->vertices.row(mesh_ptr->facets(i, 0)); // 顶点1
			Eigen::Vector3d vertex1 = mesh_ptr->vertices.row(mesh_ptr->facets(i, 1)); // 顶点2
			Eigen::Vector3d vertex2 = mesh_ptr->vertices.row(mesh_ptr->facets(i, 2)); // 顶点3
			// 法向量
			Eigen::Vector3d normal = N_faces.row(i);
			// 中心点
			Eigen::Vector3d centroid = (vertex0 + vertex1 + vertex2) / 3.0;
			pcl::PointXYZINormal point_normal;
			point_normal.x = centroid(0);
			point_normal.y = centroid(1);
			point_normal.z = centroid(2);
			point_normal.normal_x = normal(0);
			point_normal.normal_y = normal(1);
			point_normal.normal_z = normal(2);
			cloud_normal->push_back(point_normal);
		} */

		// pcl::transformPointCloudWithNormals(*cloud_normal, *cloud_normal, T_m2w.matrix());
		*cloud_normal_all += *cloud_normal;
		cout << "cloud_normal.size : " << cloud_normal->size() << endl;
	}
	double voxel_grid_size = 0.1;
	// VoxelGrid Filter
	//  pcl::VoxelGrid<pcl::PointXYZINormal> downSizeFilterMap;
	//  downSizeFilterMap.setLeafSize(voxel_grid_size, voxel_grid_size, voxel_grid_size);
	//  downSizeFilterMap.setInputCloud(cloud_normal_all);
	//  downSizeFilterMap.filter(*cloud_normal_all);
	// UniformSampling
	pcl::UniformSampling<pcl::PointXYZINormal> uniform_sampling; // 下采样滤波模型
	uniform_sampling.setInputCloud(cloud_normal_all);						 // 模型点云
	uniform_sampling.setRadiusSearch(voxel_grid_size);					 // 模型点云下采样滤波搜索半径
	uniform_sampling.filter(*cloud_normal_all);									 // 下采样得到的关键点
	cout << "dowmsamping cloud_normal.size : " << cloud_normal_all->size() << endl;
	cout << "models have inited. cost time(ms):" << t_model_init.toc() << endl;
	ros::Rate rate(10);
	while (ros::ok())
	{
		rate.sleep();
		publishCloud(pub_model_cloud_normal, cloud_normal_all, ros::Time::now(), map_frame);
	}
	return true;
}
int main(int argc, char **argv)
{
	ros::init(argc, argv, "ref_provider");
	ros::NodeHandle nh;
	ros::MultiThreadedSpinner spinner(2);
	ROS_INFO("\033[1;31m **********Point-Mesh Thread Started********** \033[0m");
	string json_path = ros::package::getPath("ref_provider") + "/config/params.json";
	std::thread m_init(modelInit, json_path); 
	// if (!modelInit(json_path))
	// 	std::cerr << "model init error" << std::endl;
	nh.param<double>("ref_provider/track_bbox_scale", track_bbox_scale, 1.0);
	nh.param<int>("ref_provider/segment_num_threshold", segment_num_threshold, 500);
	nh.param<int>("ref_provider/max_opti_num", max_opti_num, 5);
	nh.param<float>("ref_provider/search_radius", search_radius, 100.0);
	nh.param<double>("ref_provider/fitness_threshold", fitness_threshold, 0.02);
	nh.param<double>("ref_provider/max_correspondence_dist", max_correspondence_dist, 1);
	nh.param<string>("ref_provider/map_frame", map_frame, "map");
	nh.param<string>("ref_provider/point_cloud_topic", point_cloud_topic, "submap");
	nh.param<string>("ref_provider/aftPgo_odom_topic", aftPgo_odom_topic, "aftPgo_odom");
	nh.param<string>("ref_provider/model_poses_topic", model_poses_topic, "model_poses");

	std::shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> subOdom;
	std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> subCloud;
	typedef message_filters::sync_policies::ExactTime<nav_msgs::Odometry, sensor_msgs::PointCloud2> SyncPolicy;
	std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> odom_cloud_sync;
	subOdom = std::make_shared<message_filters::Subscriber<nav_msgs::Odometry>>(nh, aftPgo_odom_topic, 1000);
	subCloud = std::make_shared<message_filters::Subscriber<sensor_msgs::PointCloud2>>(nh, point_cloud_topic, 1000);
	odom_cloud_sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(1000), *subOdom, *subCloud); // 10是消息队列长度
	odom_cloud_sync->registerCallback(boost::bind(&syncedCallback, _1, _2));
	ros::Subscriber subModelPoses = nh.subscribe<nav_msgs::Path>(model_poses_topic, 1000, modelPosesCallback); // 先验模型优化位姿
	ros::Subscriber subgroundtruth = nh.subscribe<nav_msgs::Odometry>("base_pose_ground_truth", 1000, ground_truthcallback);

	pub_prior_constraint = nh.advertise<geometry_msgs::PoseWithCovariance>("prior_constraints", 1000); // 先验模型约束
	pub_prior_lidar_pts = nh.advertise<sensor_msgs::PointCloud2>("prior_lidar_pts", 1000);
	pub_prior_mesh_pts_aligned = nh.advertise<sensor_msgs::PointCloud2>("prior_mesh_pts_aligned", 1000);
	pub_prior_mesh_pts_origin = nh.advertise<sensor_msgs::PointCloud2>("prior_mesh_pts_origin", 1000);
	ground_truthpub = nh.advertise<nav_msgs::Path>("ground_truth", 10, true);
	pub_prior_constraint_edge = nh.advertise<visualization_msgs::MarkerArray>("prior_constraints_marker", 1000);
	pub_model_pose = nh.advertise<nav_msgs::Odometry>("prior_model_pose", 1000);
	pub_model_cloud_normal = nh.advertise<sensor_msgs::PointCloud2>("prior_mesh_cloud_normal", 1000);
	pub_ground_cloud = nh.advertise<sensor_msgs::PointCloud2>("ground_plane_cloud", 1000);
	
	sleep(3);
	spinner.spin();
	m_init.join();
	return 0;
}
