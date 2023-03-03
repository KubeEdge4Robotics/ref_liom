#ifndef MODEL_HPP
#define MODEL_HPP

#include "common.h"
// ceres 相关头文件
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "pcl_utils.hpp"
// igl
#include <igl/opengl/glfw/Viewer.h>
#include <igl/AABB.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/per_face_normals.h>
#include <igl/random_points_on_mesh.h>
struct PointMeshFactor
{
	PointMeshFactor(const Eigen::Vector3d &lidar_pt_, const Eigen::Vector3d &mesh_pt_)
			: lidar_pt(lidar_pt_), mesh_pt(mesh_pt_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_1{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_1{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> curr_pt{T(lidar_pt.x()), T(lidar_pt.y()), T(lidar_pt.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_1 * curr_pt + t_1;

		residual[0] = point_w.x() - T(mesh_pt.x());
		residual[1] = point_w.y() - T(mesh_pt.y());
		residual[2] = point_w.z() - T(mesh_pt.z());
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d &lidar_pt_, const Eigen::Vector3d &mesh_pt_)
	{
		return (new ceres::AutoDiffCostFunction<
						PointMeshFactor, 3, 4, 3>(
				new PointMeshFactor(lidar_pt_, mesh_pt_)));
	}

	Eigen::Vector3d lidar_pt, mesh_pt;
};

struct Mesh
{
	Eigen::MatrixXd vertices; // vertices : list of vertex positions 顶点集
	Eigen::MatrixXi facets;		// facets : list of mesh primitives(vetex indices of one facet)一个面片包含的顶点索引
														// 一行代表一个面片包含的顶点索引
	Eigen::Vector3f min_pt;		// Mesh bounding box size
	Eigen::Vector3f max_pt;
};
// Oriented Bounding Box
class OBB
{
public:
	Eigen::Vector3f min_pt; // min_(x,y,z)
	Eigen::Vector3f max_pt; // max_(x,y,z)
	// world frame to model frame
	Eigen::Vector3f rot;	 //(roll, pitch, yaw)
	Eigen::Vector3f trans; //(x,y,z)
	// default constructor
	OBB() : min_pt(0, 0, 0), max_pt(0, 0, 0), rot(0, 0, 0), trans(0, 0, 0){};
};
// 先验模型
class PriorModel
{
public:
	std::string name;
	OBB bbox;
	std::shared_ptr<igl::AABB<Eigen::MatrixXd, 3>> tree_ptr;
	std::shared_ptr<Mesh> mesh_ptr;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr;
	std::unordered_map<VOXEL_LOC, int> hash_map3d;
	PriorModel() : bbox(), mesh_ptr(nullptr), tree_ptr(nullptr), cloud_ptr(nullptr){};
	PriorModel(const std::shared_ptr<Mesh> &m) : mesh_ptr(m), tree_ptr(new igl::AABB<Eigen::MatrixXd, 3>), cloud_ptr(nullptr)
	{
		tree_ptr->init(mesh_ptr->vertices, mesh_ptr->facets);
	};
};

class Point2meshICP
{
protected:
	double max_correspondence_dist_;
	double fitness_;
	int outlier_num_;
	Eigen::Affine3d transform_;
	double outlier_threshold_;
	bool converged_;
	bool set_rot_constant_;

public:
	// constructor
	Point2meshICP() : max_correspondence_dist_(0.5), fitness_(std::numeric_limits<double>::max()),
										outlier_num_(0), transform_(Eigen::Affine3d::Identity()), outlier_threshold_(1.0),
										converged_(false), set_rot_constant_(false){};
	Point2meshICP(double max_dist) : max_correspondence_dist_(0.5), fitness_(std::numeric_limits<double>::max()),
																	 outlier_num_(0), transform_(Eigen::Affine3d::Identity()), outlier_threshold_(1.0),
																	 converged_(false), set_rot_constant_(false){};
	void setRotCostant()
	{
		set_rot_constant_ = true;
	}
	void setRotVariable()
	{
		set_rot_constant_ = false;
	}
	// set max outliers proportion
	void setOutlierThreshold(double threshold)
	{
		outlier_threshold_ = threshold;
	}
	// set max distance between correspondences
	void setMaxCorrespondenceDist(double max_dist)
	{
		max_correspondence_dist_ = max_dist;
	}
	// solve one point-mesh ICP
	bool operator()(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_segmented_model, const std::shared_ptr<PriorModel> &prior_model, pcl::PointCloud<pcl::PointXYZ>::Ptr &mesh_cloud, pcl::IndicesPtr outlier_indices = nullptr)
	{
		converged_ = false; // 重置收敛标志
		outlier_num_ = 0;		// 重置outlier_num_
		if (outlier_indices)
			outlier_indices->clear();
		// 搜索虚拟点+点云网格配准
		Eigen::VectorXd sqrD;						// list of squared distances
		Eigen::VectorXi indices_facets; // list of indices into Element of closest mesh primitive
		Eigen::MatrixXd closest_points; // list of closest points
		// ceres ICP
		Eigen::MatrixXd query_points(cloud_segmented_model->size(), 3); // query_points : list of query points
		for (int i = 0; i < cloud_segmented_model->size(); i++)
		{
			query_points(i, 0) = cloud_segmented_model->points[i].x;
			query_points(i, 1) = cloud_segmented_model->points[i].y;
			query_points(i, 2) = cloud_segmented_model->points[i].z;
		}
		// 更新mesh匹配
		mesh_cloud->clear();
		TicToc t_point_mesh; // 搜索匹配点的时间
		prior_model->tree_ptr->squared_distance(prior_model->mesh_ptr->vertices, prior_model->mesh_ptr->facets, query_points, sqrD, indices_facets, closest_points);
		std::cout << "*******************************" << std::endl;
		std::cout << "prior_lidar_pts size: " << cloud_segmented_model->size() << "  "
							<< "point-mesh time (ms): " << t_point_mesh.toc() << std::endl;
		uint32_t meshnum = closest_points.rows(); // mesh虚拟点数目
		for (uint32_t i = 0; i < meshnum; i++)
		{
			pcl::PointXYZ pointc; // 匹配最近点
			pointc.x = closest_points(i, 0);
			pointc.y = closest_points(i, 1);
			pointc.z = closest_points(i, 2);
			mesh_cloud->points.push_back(pointc);
		}
		double scale_mesh = 1.0 / meshnum; // point-to-mesh权重系数
		// 构造损失函数
		double parameter[7] = {0, 0, 0, 1, 0, 0, 0}; // 迭代初值
		Eigen::Map<Eigen::Quaterniond> q(parameter);
		Eigen::Map<Eigen::Vector3d> t(parameter + 4);
		ceres::LossFunction *loss_function = new ceres::CauchyLoss(0.1); // ceres::HuberLoss(0.1);
		ceres::LossFunction *loss_function_scaled =
				new ceres::ScaledLoss(loss_function, scale_mesh, ceres::Ownership::DO_NOT_TAKE_OWNERSHIP);
		ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
		ceres::Problem::Options problem_options;
		ceres::Problem problem(problem_options);
		problem.AddParameterBlock(parameter, 4, q_parameterization); // 添加参数块，旋转四元数
		problem.AddParameterBlock(parameter + 4, 3);
		if (set_rot_constant_)
		{
			problem.SetParameterBlockConstant(parameter);
		}
		else
		{
			problem.SetParameterBlockVariable(parameter);
		}
		for (int i = 0; i < meshnum; i++)
		{
			pcl::PointXYZ &lidarPt = cloud_segmented_model->points[i];
			pcl::PointXYZ &meshPt = mesh_cloud->points[i];
			Eigen::Vector3d lidar_pt(lidarPt.x, lidarPt.y, lidarPt.z);
			Eigen::Vector3d mesh_pt(meshPt.x, meshPt.y, meshPt.z);
			ceres::CostFunction *cost_function = PointMeshFactor::Create(lidar_pt, mesh_pt);
			problem.AddResidualBlock(cost_function, loss_function_scaled, parameter, parameter + 4);
			if (sqrt(sqrD[i]) > max_correspondence_dist_)
			{
				// 异常值 + 1
				outlier_num_++;
				if (outlier_indices)
					outlier_indices->push_back(i);
			}
		}
		double outlier_proportion = static_cast<double>(outlier_num_) / static_cast<double>(cloud_segmented_model->size());
		// if (outlier_proportion > outlier_threshold_)
		// 	return false;
		TicToc t_solver;
		ceres::Solver::Options options;
		options.minimizer_type = ceres::TRUST_REGION;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // ceres::DENSE_QR;
		options.dense_linear_algebra_library_type = ceres::LAPACK;
		options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
		options.max_num_iterations = 100;
		options.minimizer_progress_to_stdout = false;
		options.check_gradients = false;
		options.gradient_check_relative_precision = 1e-4;
		options.parameter_tolerance = 1e-8;
		options.function_tolerance = 1e-8;
		options.gradient_tolerance = 1e-12;
		options.max_solver_time_in_seconds = 0.5;
		options.num_threads = 8;
		options.initial_trust_region_radius = 1e4;											 // 初始信任区域的大小。
		options.max_trust_region_radius = 1e20;													 // 信任区域半径最大值。
		options.min_trust_region_radius = 1e-32;												 // 信任区域的最小值。当信任区域小于此值，会停止优化。
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT; // 还有ceres::DOGLEG
		// options.dogleg_type = ceres::SUBSPACE_DOGLEG // 使用DOGLEG方法时指定
		// options.min_relative_decrease = ;//信任域步长(trust region step)相对减少的最小值。
		// options.min_lm_diagonal = ;//LEVENBERG MARQUARDT算法使用对角矩阵来规范（regularize）信任域步长。 这是该对角矩阵的值的下限
		// options.max_lm_diagonal = ;//LEVENBERG MARQUARDT算法使用对角矩阵来规范（regularize）信任域步长。这是该对角矩阵的值的上限。
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		fitness_ = summary.final_cost; // 最终函数值
		// 结果输出
		std::cout << "fitness : " << fitness_ << "  ";
		std::cout << "temination_type:" << summary.termination_type << "  ";							 // 0:CONVERGENCE, 1:NO_CONVERGENCE, 3:USER_SUCCESS
		std::cout << "num_successful_steps:" << summary.num_successful_steps << std::endl; // 最终被接受的迭代次数
		std::cout << "*******************************" << std::endl;
		if (summary.termination_type == 1)
		{
			converged_ = false;
			return false;
		}
		else
			converged_ = true;
		// std::cout << summary.BriefReport() << std::endl;
		// 优化结果
		transform_.setIdentity();
		transform_.rotate(q);
		transform_.pretranslate(t);
		return true;
	}
	inline double getFitness()
	{
		return (fitness_);
	}
	inline Eigen::Affine3d getTransform()
	{
		return (transform_);
	}
	inline bool hasConverged()
	{
		return (converged_);
	}
};
std::shared_ptr<PriorModel> modelInit(const std::string &name, const std::shared_ptr<Mesh> &mesh_ptr,
																			const Eigen::Affine3d &T_w2m, double voxel_size = 0.1)
{
	std::shared_ptr<PriorModel> model_ptr(new PriorModel(mesh_ptr));
	double x, y, z, roll, pitch, yaw;
	pcl::getTranslationAndEulerAngles(T_w2m, x, y, z, roll, pitch, yaw);
	model_ptr->bbox.trans << x, y, z;
	model_ptr->bbox.rot << roll, pitch, yaw;
	// mesh to point cloud with normal vector
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normal(new pcl::PointCloud<pcl::Normal>);
	// Eigen::MatrixXd N_vertices;
	// // Compute per-vertex normals
	// igl::per_vertex_normals(mesh_ptr->vertices, mesh_ptr->facets, N_vertices);
	// Eigen::MatrixXd N_faces;
	// // Compute per-face normals
	// igl::per_face_normals(mesh_ptr->vertices, mesh_ptr->facets, N_faces);

	// add vertex
	for (int i = 0; i < mesh_ptr->vertices.rows(); i++)
	{
		pcl::PointXYZ point;
		point.x = mesh_ptr->vertices(i, 0);
		point.y = mesh_ptr->vertices(i, 1);
		point.z = mesh_ptr->vertices(i, 2);
		cloud_points->push_back(point);
		/* // 法向量
		pcl::Normal normal;
		normal.normal_x = N_vertices(i, 0);
		normal.normal_y = N_vertices(i, 1);
		normal.normal_z = N_vertices(i, 2);
		cloud_normal->push_back(normal); */
	}

	// samping on facets
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
		pcl::PointXYZ point;
		point.x = vertex(0);
		point.y = vertex(1);
		point.z = vertex(2);
		cloud_points->push_back(point);
		/* // 法向量
		Eigen::Vector3d face_normal = N_faces.row(F_index(i));
		pcl::Normal normal;
		normal.normal_x = face_normal(0);
		normal.normal_y = face_normal(1);
		normal.normal_z = face_normal(2);
		cloud_normal->push_back(normal); */
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
	model_ptr->cloud_ptr = cloud_points;
	Eigen::Vector4f min_pt;
	Eigen::Vector4f max_pt;
	pcl::getMinMax3D(*cloud_points, min_pt, max_pt);
	model_ptr->bbox.min_pt = min_pt.head<3>();
	model_ptr->bbox.max_pt = max_pt.head<3>();
	cutVoxel3d<pcl::PointXYZ>(model_ptr->hash_map3d, cloud_points, voxel_size);
	return model_ptr;
}

bool modelRegister(
		std::shared_ptr<PriorModel> &prior_model,
		Point2meshICP &point2model_icp, Eigen::Affine3d &T_b2m,
		pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_segmented_body,
		pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_others,
		pcl::PointCloud<pcl::PointXYZ>::Ptr &model_cloud,
		int segment_num_threshold = 300,
		double fitness_threshold = 0.01, double max_correspondence_dist = 0.4,
		double voxel_size = 0.05, double overlap_threshold = 0.5,
		int max_opti_num = 10)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented_model(new pcl::PointCloud<pcl::PointXYZ>); // 分割得到的点云(model系)
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	cout << "setRotConstant" << endl;
	// 固定旋转，平移粗配准
	point2model_icp.setRotCostant(); // 设置旋转不变
	// 设置最大匹配距离
	point2model_icp.setMaxCorrespondenceDist(0.5 * (prior_model->bbox.max_pt - prior_model->bbox.min_pt).norm());
	pcl::IndicesPtr outlier_indices(new pcl::Indices);
	double last_fitness = std::numeric_limits<double>::max(); // 上一次匹配目标函数值
	for (int i = 0; i < max_opti_num; i++)
	{
		// 如果分割点云过少，则跳过当前循环
		if (cloud_segmented_body->size() < segment_num_threshold)
		{
			cout << "too few segmented points: " << cloud_segmented_body->size() << endl;
			return false;
		}
		// 更新原始点云world_to_model
		pcl::transformPointCloud(*cloud_segmented_body, *cloud_segmented_model, T_b2m); // 更新之后的模型系点云
		// point-mesh icp
		if (!point2model_icp(cloud_segmented_model, prior_model, model_cloud, outlier_indices))
		{
			// 如果异常点过多，调整缩放因子，再次分割
			cout << "outlier too much" << endl;
			// shrink_scale *= 0.9;
			continue;
		}
		// 只有当收敛了才会更新
		if (point2model_icp.hasConverged() && point2model_icp.getFitness() < last_fitness)
		{
			if (fabs(last_fitness - point2model_icp.getFitness()) < 1e-8)
			{
				cout << "break" << endl;
				break;
			}
			last_fitness = point2model_icp.getFitness();
			Eigen::Affine3d T_inc = point2model_icp.getTransform(); // 位姿增量
			T_b2m = T_inc * T_b2m;																	// 更新T_w2m
			// ICP迭代结束，点云发布
			std::cout << "ICP update finish" << std::endl;
		}
		// else
		// {
		// 	// 如果ICP结果不好，调整，重新ICP
		// 	shrink_scale *= 0.9;
		// }
		if (point2model_icp.getFitness() < fitness_threshold)
		{
			// 将分割剩余点云放回pcl_cloud_body供下一个模型提取
			pcl::PointCloud<pcl::PointXYZ> cloud_add;
			extract.setInputCloud(cloud_segmented_body);
			extract.setIndices(outlier_indices);
			extract.setNegative(false); // 设置为提取索引提供的点云
			extract.filter(cloud_add);
			*cloud_others += cloud_add; // 添加分割剩余点云
			// 提取分割后点云(body系)
			extract.setInputCloud(cloud_segmented_body);
			extract.setIndices(outlier_indices);
			extract.setNegative(true); // 设置为排除索引提供的点云
			extract.filter(*cloud_segmented_body);
			if (outlier_indices->size() < 10)
			{
				cout << "break" << endl;
				break;
			}
			// 设置最大匹配距离
			// point2model_icp.setMaxCorrespondenceDist(max_correspondence_dist);
		}
	}
	cout << "setRotaVariable" << endl;
	// 同时优化旋转与平移，得到精细配准结果
	point2model_icp.setRotVariable(); // 恢复旋转为变量
	// 设置最大匹配距离
	point2model_icp.setMaxCorrespondenceDist(max_correspondence_dist);
	for (int i = 0; i < max_opti_num; i++)
	{
		// 如果分割点云过少，则跳过当前循环
		if (cloud_segmented_body->size() < segment_num_threshold)
		{
			cout << "too few segmented points: " << cloud_segmented_body->size() << endl;
			return false;
		}
		// 更新原始点云world_to_model
		pcl::transformPointCloud(*cloud_segmented_body, *cloud_segmented_model, T_b2m); // 更新之后的模型系点云
		// point-mesh icp
		if (!point2model_icp(cloud_segmented_model, prior_model, model_cloud, outlier_indices))
		{
			// 如果异常点过多，调整缩放因子，再次分割
			cout << "outlier too much" << endl;
			// shrink_scale *= 0.9;
			continue;
		}
		// 只有当收敛了才会更新
		if (point2model_icp.hasConverged() && point2model_icp.getFitness() < last_fitness)
		{
			if (fabs(last_fitness - point2model_icp.getFitness()) < 1e-8)
			{
				cout << "break" << endl;
				break;
			}
			last_fitness = point2model_icp.getFitness();
			Eigen::Affine3d T_inc = point2model_icp.getTransform(); // 位姿增量
			T_b2m = T_inc * T_b2m;																	// 更新T_w2m
			// ICP迭代结束，点云发布
			std::cout << "ICP update finish" << std::endl;
		}
		// else
		// {
		// 	// 如果ICP结果不好，调整，重新ICP
		// 	shrink_scale *= 0.9;
		// }
		/* if (point2model_icp.getFitness() < fitness_threshold)
		{
			// 将分割剩余点云放回pcl_cloud_body供下一个模型提取
			pcl::PointCloud<pcl::PointXYZ> cloud_add;
			extract.setInputCloud(cloud_segmented_body);
			extract.setIndices(outlier_indices);
			extract.setNegative(false); // 设置为提取索引提供的点云
			extract.filter(cloud_add);
			*cloud_others += cloud_add; // 添加分割剩余点云
			// 提取分割后点云(body系)
			extract.setInputCloud(cloud_segmented_body);
			extract.setIndices(outlier_indices);
			extract.setNegative(true); // 设置为排除索引提供的点云
			extract.filter(*cloud_segmented_body);
			if (outlier_indices->size() < 10)
				break;
			// 设置最大匹配距离
			point2model_icp.setMaxCorrespondenceDist(sqrt(point2model_icp.getFitness() / fitness_threshold) * max_correspondence_dist);
		} */
	}

	/* // 更新原始点云world_to_model
	pcl::transformPointCloud(*cloud_segmented_body, *cloud_segmented_model, T_b2m); // 更新之后的模型系点云
	// 计算重合度，同时剔除杂点
	pcl::PointCloud<pcl::PointXYZ> cloud_add;
	cout << "calculateOverlapScore One stage" << endl;
	float overlap_score = calculateOverlapScore<pcl::PointXYZ>(prior_model->cloud_ptr, cloud_segmented_model, cloud_add, 10 * voxel_size, 0.5);
	cout << "one stage overlap score: " << overlap_score << endl;
	if (overlap_score < overlap_threshold)
	{
		cout << "overlap score too low" << endl;
		return false;
	}
	*cloud_others += cloud_add;
	// 进一步精细优化
	for (int i = 0; i < max_opti_num; i++)
	{
		// 如果分割点云过少，则跳过当前循环
		if (cloud_segmented_model->size() < segment_num_threshold)
		{
			cout << "too few segmented points: " << cloud_segmented_model->size() << endl;
			return false;
		}
		// point-mesh icp
		if (!point2model_icp(cloud_segmented_model, prior_model, model_cloud, outlier_indices))
		{
			// 如果异常点过多，调整缩放因子，再次分割
			cout << "outlier too much" << endl;
			// shrink_scale *= 0.9;
			continue;
		}
		// 只有当收敛了才会更新
		if (point2model_icp.hasConverged() && point2model_icp.getFitness() < last_fitness)
		{
			last_fitness = point2model_icp.getFitness();
			Eigen::Affine3d T_inc = point2model_icp.getTransform(); // 位姿增量
			// ICP迭代结束，点云发布
			std::cout << "ICP update finish" << std::endl;
			// 更新model系点云
			pcl::transformPointCloud(*cloud_segmented_model, *cloud_segmented_model, T_inc); // 更新之后的模型系点云
		}
	} */

	return true;
}
#endif // !MODEL_HPP
