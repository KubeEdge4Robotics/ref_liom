#ifndef PCL_UTILS_HPP
#define PCL_UTILS_HPP
#include <vector>
#include <list>
#include <iostream>
#include <thread>
#include <chrono>
// pcl 头文件
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
// 转换
#include <pcl/common/transforms.h> //点云转换 转换矩阵
#include <pcl/common/pca.h>
#include <pcl/common/eigen.h>
// 特征
#include <pcl/features/normal_3d.h>			//法向量特征
#include <pcl/features/normal_3d_omp.h> //法向量特征
// 文件操作
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/io.h>
#include <pcl/io/vtk_lib_io.h>
// 可视化
#include <pcl/visualization/pcl_visualizer.h> //可视化
// 配准
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
// kdtree
#include <pcl/kdtree/kdtree_flann.h> // kdtree 快速近邻搜索
#include <pcl/kdtree/impl/kdtree_flann.hpp>
// 滤波的头文件
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/uniform_sampling.h>						 //均匀采样 滤波
#include <pcl/filters/crop_box.h>										 //分割bounding box头文件
#include <pcl/filters/statistical_outlier_removal.h> //统计滤波 头文件
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
// ransac
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h> //SampleConsensusModelPlane
#include <pcl/sample_consensus/ransac.h>					//RandomSampleConsensus
#include <pcl/segmentation/extract_clusters.h>		//EuclideanClusterExtraction
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/recognition/ransac_based/obj_rec_ransac.h>
//
//========计算法线向量==============
//
template <typename PointT>
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(typename pcl::PointCloud<PointT>::Ptr points_in, double radius)
{
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimationOMP<PointT, pcl::Normal> norm_est; // 多核 计算法线模型 OpenMP
	// pcl::NormalEstimation<PointT, pcl::Normal> norm_est; // 单核计算法线模型
	norm_est.setRadiusSearch(radius);
	// norm_est.setKSearch(10); // 最近10个点 协方差矩阵PCA分解 计算 法线向量
	norm_est.setNumberOfThreads(8);
	typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
	norm_est.setSearchMethod(tree);		 // 多核模式 不需要设置 搜索算法
	norm_est.setInputCloud(points_in); // 模型点云
	norm_est.compute(*normals);				 // 模型点云的法线向量
	return normals;
}

void segmentPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input, pcl::Indices &indices_plane,
									int max_iterations = 1000, double threshold = 0.05)
{
	// Estimate
	using SampleConsensusModelPlanePtr = pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr;
	SampleConsensusModelPlanePtr model(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(input));
	pcl::RandomSampleConsensus<pcl::PointXYZ> sac(model, threshold);
	sac.setMaxIterations(max_iterations);
	bool res = sac.computeModel();

	sac.getInliers(indices_plane);
	Eigen::VectorXf coefficients;
	sac.getModelCoefficients(coefficients);

	if (!res || indices_plane.empty())
	{
		PCL_ERROR("No planar model found. Relax thresholds and continue.\n");
		return;
	}
	// Refine the plane indices
	sac.refineModel(2, 50);
	sac.getInliers(indices_plane);
	sac.getModelCoefficients(coefficients);
}

// Detect the largest plane and remove it from the sets
void segmentPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input, pcl::PointCloud<pcl::PointXYZ> &output,
									bool negative = false, int max_iterations = 1000, double threshold = 0.05)
{
	// Estimate
	using SampleConsensusModelPlanePtr = pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr;
	SampleConsensusModelPlanePtr model(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(input));
	pcl::RandomSampleConsensus<pcl::PointXYZ> sac(model, threshold);
	sac.setMaxIterations(max_iterations);
	bool res = sac.computeModel();

	pcl::IndicesPtr indices_plane(new pcl::Indices);
	sac.getInliers(*indices_plane);
	Eigen::VectorXf coefficients;
	sac.getModelCoefficients(coefficients);

	if (!res || indices_plane->empty())
	{
		PCL_ERROR("No planar model found. Relax thresholds and continue.\n");
		return;
	}
	// Refine the plane indices
	sac.refineModel(2, 50);
	sac.getInliers(*indices_plane);
	sac.getModelCoefficients(coefficients);
	if (negative)
	{
		// 提取非平面点云
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(input);
		extract.setIndices(indices_plane);
		extract.setNegative(true); // 设置为剔除索引提供的点云
		extract.filter(output);
	}
	else
	{
		// 提取平面点云
		pcl::copyPointCloud(*input, *indices_plane, output);
	}
}

/* // Detect the largest plane and remove it from the sets
template <typename PointT>
class SegPlane
{
protected:
	pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg; // 依据法线　分割对象
public:
	typename pcl::PointCloud<PointT>::Ptr non_plane_points;
	pcl::PointCloud<pcl::Normal>::Ptr non_plane_normals;
	typename pcl::PointCloud<PointT>::Ptr plane_points;
	SegPlane(double dist_threshold) : plane_points(new pcl::PointCloud<PointT>), non_plane_points(new pcl::PointCloud<PointT>)
	{
		// Create the segmentation object
		seg.setOptimizeCoefficients(true);						// 同时优化模型系数
		seg.setModelType(pcl::SACMODEL_NORMAL_PLANE); // 平面模型
		seg.setMethodType(pcl::SAC_RANSAC);						// 随机采样一致性算法
		seg.setMaxIterations(100);										// 最大迭代次数
		seg.setNormalDistanceWeight(0.1);							// 法线信息权重
		seg.setDistanceThreshold(dist_threshold);			// 设置内点到模型的距离允许最大值
		// seg.setNumberOfThreads(8);
	}
	bool segment(typename pcl::PointCloud<PointT>::Ptr all_points, pcl::PointCloud<pcl::Normal>::Ptr all_normals)
	{
		pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients());
		pcl::PointIndices::Ptr indices_plane(new pcl::PointIndices());

		seg.setInputCloud(all_points);		// 输入点云
		seg.setInputNormals(all_normals); // 输入法线特征
		// 获取平面模型的系数和处在平面的内点
		seg.segment(*indices_plane, *coefficients_plane); // 分割　得到内点索引　和模型系数

		if (indices_plane->indices.empty())
		{
			PCL_ERROR("Could not estimate a planar model for the given dataset.");
			return false;
		}
		else
		{
			std::cout << "Plane coefficients: " << *coefficients_plane << std::endl;
		}
		pcl::ExtractIndices<PointT> extract;
		pcl::ExtractIndices<pcl::Normal> extract_normals;
		// 提取平面点云
		extract.setInputCloud(all_points);
		extract.setIndices(indices_plane);
		extract.setNegative(false); // 设置为提取索引提供的点云
		extract.filter(*plane_points);
		// 提取非平面点云
		extract.setInputCloud(all_points);
		extract.setIndices(indices_plane);
		extract.setNegative(true); // 设置为排除索引提供的点云
		extract.filter(*non_plane_points);
		extract_normals.setInputCloud(all_normals);
		extract_normals.setIndices(indices_plane);
		extract_normals.setNegative(true); // 设置为提取索引提供的点云
		extract_normals.filter(*non_plane_normals);
		return true;
	}
}; */

void arrayToVtkMatrix(const float *a, vtkMatrix4x4 *m)
{
	// Setup the rotation
	m->SetElement(0, 0, a[0]);
	m->SetElement(0, 1, a[1]);
	m->SetElement(0, 2, a[2]);
	m->SetElement(1, 0, a[3]);
	m->SetElement(1, 1, a[4]);
	m->SetElement(1, 2, a[5]);
	m->SetElement(2, 0, a[6]);
	m->SetElement(2, 1, a[7]);
	m->SetElement(2, 2, a[8]);
	// Setup the translation
	m->SetElement(0, 3, a[9]);
	m->SetElement(1, 3, a[10]);
	m->SetElement(2, 3, a[11]);
}
using namespace pcl::recognition;

class Recognizer
{
	std::shared_ptr<pcl::recognition::ObjRecRANSAC> objrec_; // The object recognizer
	pcl::visualization::PCLVisualizer viz_;
	std::shared_ptr<std::thread> visualize_thread;
	int num_hypotheses_to_show_;

public:
	using Hypothesis = pcl::recognition::Hypothesis;
	using ObjRecRANSAC = pcl::recognition::ObjRecRANSAC;
	enum class ObjRecMode
	{
		TEST_MODE = 0,
		FULL_MODE = 1
	};
	Recognizer(float pair_width, float voxel_size, float max_coplanarity_angle) : objrec_(new ObjRecRANSAC(pair_width, voxel_size)),
																																								visualize_thread(new std::thread(&Recognizer::visualizeLoop, this)),
																																								num_hypotheses_to_show_(1)
	{
		objrec_->setMaxCoplanarityAngleDegrees(max_coplanarity_angle);
	}
	void addModel(pcl::PointCloud<pcl::PointXYZ>::Ptr model_points, pcl::PointCloud<pcl::Normal>::Ptr model_normals, const std::string &model_name)
	{
		// Add the model
		objrec_->addModel(*model_points, *model_normals, model_name);
	}
	bool recognize(pcl::PointCloud<pcl::PointXYZ>::Ptr scene_points, pcl::PointCloud<pcl::Normal>::Ptr scene_normals, ObjRecMode mode)
	{
		// This will be the output of the recognition
		std::list<ObjRecRANSAC::Output> rec_output;
		if (mode == ObjRecMode::FULL_MODE)
		{
			objrec_->leaveTestMode();
			// Run the recognition method
			objrec_->recognize(*scene_points, *scene_normals, rec_output);
			if (rec_output.size() > 0)
			{
				// Sort the hypotheses vector such that the strongest hypotheses are at the front
				rec_output.sort([](const ObjRecRANSAC::Output &a, const ObjRecRANSAC::Output &b)
												{ return a.match_confidence_ > b.match_confidence_; });
				// Show the output result
				int i = 0;
				for (auto it = rec_output.begin(); i < num_hypotheses_to_show_ && it != rec_output.end(); ++i, ++it)
				{
					// Visualize the orientation as a tripod
					char frame_name[128];
					sprintf(frame_name, "frame_%i", i + 1);
					showHypothesisAsCoordinateFrame(*it, frame_name);
					// Compose the model's id
					std::cout << it->object_name_ << "_" << i + 1 << " has a confidence value of " << it->match_confidence_ << ";  ";
				}
			}
			else
				return false;
		}
		else if (mode == ObjRecMode::TEST_MODE)
		{
			// Switch to the test mode in which only oriented point pairs from the scene are sampled
			objrec_->enterTestModeTestHypotheses();
			// Run the recognition method
			objrec_->recognize(*scene_points, *scene_normals, rec_output);
			// Now show some of the accepted hypotheses
			std::vector<Hypothesis> accepted_hypotheses;
			objrec_->getAcceptedHypotheses(accepted_hypotheses);
			if (accepted_hypotheses.size() > 0)
			{
				// Sort the hypotheses vector such that the strongest hypotheses are at the front
				std::sort(accepted_hypotheses.begin(), accepted_hypotheses.end(),
									[](const Hypothesis &a, const Hypothesis &b)
									{ return a.match_confidence_ > b.match_confidence_; });

				// Show the hypotheses
				int i = 0;
				for (auto acc_hypo = accepted_hypotheses.begin(); i < num_hypotheses_to_show_ && acc_hypo != accepted_hypotheses.end(); ++i, ++acc_hypo)
				{
					// Visualize the orientation as a tripod
					char frame_name[128];
					sprintf(frame_name, "frame_%i", i + 1);
					showHypothesisAsCoordinateFrame(*acc_hypo, frame_name);
					// Compose the model's id
					std::cout << acc_hypo->obj_model_->getObjectName() << "_" << i + 1 << " has a confidence value of " << acc_hypo->match_confidence_ << ";  ";
				}
			}
			else
				return false;
		}

		viz_.addPointCloud(scene_points, "scene points");
		viz_.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scene points");

		pcl::PointCloud<pcl::PointXYZ>::Ptr points_octree(new pcl::PointCloud<pcl::PointXYZ>());
		objrec_->getSceneOctree().getFullLeavesPoints(*points_octree);
		// viz_.addPointCloud(points_octree, "octree points");
		// viz_.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "octree points");
		// viz_.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "octree points");

		// viz_.addPointCloud(plane_points, "plane points");
		// viz_.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.9, 0.9, 0.9, "plane points");
		pcl::PointCloud<pcl::Normal>::Ptr normals_octree(new pcl::PointCloud<pcl::Normal>());
		objrec_->getSceneOctree().getNormalsOfFullLeaves(*normals_octree);
		viz_.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(points_octree, normals_octree, 1, 6.0f, "normals out");
		return true;
	}
	void visualizeLoop()
	{
		// Enter the main loop
		while (!viz_.wasStopped())
		{
			// main loop of the visualizer
			viz_.spinOnce(100);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}
	void showHypothesisAsCoordinateFrame(Hypothesis &hypo, const std::string &frame_name)
	{
		float rot_col[3], x_dir[3], y_dir[3], z_dir[3], origin[3], scale = 2.0f * objrec_->getPairWidth();
		pcl::ModelCoefficients coeffs;
		coeffs.values.resize(6);
		// Get the origin of the coordinate frame
		aux::transform(hypo.rigid_transform_, hypo.obj_model_->getOctreeCenterOfMass(), origin);
		coeffs.values[0] = origin[0];
		coeffs.values[1] = origin[1];
		coeffs.values[2] = origin[2];
		// Setup the axes
		rot_col[0] = hypo.rigid_transform_[0];
		rot_col[1] = hypo.rigid_transform_[3];
		rot_col[2] = hypo.rigid_transform_[6];
		aux::mult3(rot_col, scale, x_dir);
		rot_col[0] = hypo.rigid_transform_[1];
		rot_col[1] = hypo.rigid_transform_[4];
		rot_col[2] = hypo.rigid_transform_[7];
		aux::mult3(rot_col, scale, y_dir);
		rot_col[0] = hypo.rigid_transform_[2];
		rot_col[1] = hypo.rigid_transform_[5];
		rot_col[2] = hypo.rigid_transform_[8];
		aux::mult3(rot_col, scale, z_dir);

		// The x-axis
		coeffs.values[3] = x_dir[0];
		coeffs.values[4] = x_dir[1];
		coeffs.values[5] = x_dir[2];
		viz_.addLine(coeffs, frame_name + "_x_axis");
		viz_.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, frame_name + "_x_axis");

		// The y-axis
		coeffs.values[3] = y_dir[0];
		coeffs.values[4] = y_dir[1];
		coeffs.values[5] = y_dir[2];
		viz_.addLine(coeffs, frame_name + "_y_axis");
		viz_.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, frame_name + "_y_axis");

		// The z-axis
		coeffs.values[3] = z_dir[0];
		coeffs.values[4] = z_dir[1];
		coeffs.values[5] = z_dir[2];
		viz_.addLine(coeffs, frame_name + "_z_axis");
		viz_.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, frame_name + "_z_axis");
	}
	void showHypothesisAsCoordinateFrame(ObjRecRANSAC::Output &rec_output, const std::string &frame_name)
	{
		float rot_col[3], x_dir[3], y_dir[3], z_dir[3], origin[3], scale = 2.0f * objrec_->getPairWidth();
		pcl::ModelCoefficients coeffs;
		coeffs.values.resize(6);
		// Get the origin of the coordinate frame
		aux::transform(rec_output.rigid_transform_, objrec_->getModel(rec_output.object_name_)->getOctreeCenterOfMass(), origin);
		coeffs.values[0] = origin[0];
		coeffs.values[1] = origin[1];
		coeffs.values[2] = origin[2];
		// Setup the axes
		rot_col[0] = rec_output.rigid_transform_[0];
		rot_col[1] = rec_output.rigid_transform_[3];
		rot_col[2] = rec_output.rigid_transform_[6];
		aux::mult3(rot_col, scale, x_dir);
		rot_col[0] = rec_output.rigid_transform_[1];
		rot_col[1] = rec_output.rigid_transform_[4];
		rot_col[2] = rec_output.rigid_transform_[7];
		aux::mult3(rot_col, scale, y_dir);
		rot_col[0] = rec_output.rigid_transform_[2];
		rot_col[1] = rec_output.rigid_transform_[5];
		rot_col[2] = rec_output.rigid_transform_[8];
		aux::mult3(rot_col, scale, z_dir);

		// The x-axis
		coeffs.values[3] = x_dir[0];
		coeffs.values[4] = x_dir[1];
		coeffs.values[5] = x_dir[2];
		viz_.addLine(coeffs, frame_name + "_x_axis");
		viz_.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, frame_name + "_x_axis");

		// The y-axis
		coeffs.values[3] = y_dir[0];
		coeffs.values[4] = y_dir[1];
		coeffs.values[5] = y_dir[2];
		viz_.addLine(coeffs, frame_name + "_y_axis");
		viz_.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, frame_name + "_y_axis");

		// The z-axis
		coeffs.values[3] = z_dir[0];
		coeffs.values[4] = z_dir[1];
		coeffs.values[5] = z_dir[2];
		viz_.addLine(coeffs, frame_name + "_z_axis");
		viz_.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, frame_name + "_z_axis");
	}
};
#endif