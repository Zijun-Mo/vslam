/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include"Atlas.h"
#include"MapPoint.h"
#include"KeyFrame.h"
#include "Settings.h"
#include<pangolin/pangolin.h>

#include<chrono>
#include<mutex>

namespace open3d { namespace geometry { class TriangleMesh; } }

namespace ORB_SLAM3
{

class Settings;

class MapDrawer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MapDrawer(Atlas* pAtlas, const string &strSettingPath, Settings* settings);

    void newParameterLoader(Settings* settings);

    Atlas* mpAtlas;

    void DrawMapPoints();
    void DrawVGGTDenseCloud(bool onlyActiveMap, size_t maxPoints, float pointSizeOverride, int refreshMs = 0);
    void DrawTSDFMesh(bool wireframe, size_t maxFaces, float lineWidth, float faceAlpha, int refreshMs = 0);
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph, const bool bDrawOptLba);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
    void SetCurrentCameraPose(const Sophus::SE3f &Tcw);
    void SetReferenceKeyFrame(KeyFrame *pKF);
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw);

    // Add a non-keyframe (or keyframe) pose marker to be visualized as trajectory points.
    // If isKeyFrame is true we can choose to draw with a different color.
    void AddFramePose(const Sophus::SE3f &Tcw, bool isKeyFrame);
    void DrawFrameTrajectory();
    void ClearFrameTrajectory();

    float GetPointSize() const { return mPointSize; }

private:

    bool ParseViewerParamFile(cv::FileStorage &fSettings);

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    Sophus::SE3f mCameraPose;

    std::mutex mMutexCamera;

    // Cached dense cloud / TSDF mesh for viewer throttling
    std::vector<VGGTDensePointRGBXYZ> mCachedDensePoints;
    std::chrono::steady_clock::time_point mLastDenseFetch;
    std::mutex mMutexDenseCache;

    std::shared_ptr<open3d::geometry::TriangleMesh> mCachedTSDFMesh;
    std::chrono::steady_clock::time_point mLastTSDFFetch;
    std::mutex mMutexTSDFCache;

    // Stored trajectory poses for non-keyframes (and optionally keyframes) in world coordinates Twc
    std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> mFrameTrajectory;
    std::vector<bool> mFrameIsKeyFrame; // parallel array marking keyframes
    std::mutex mMutexTrajectory;

    float mfFrameColors[6][3] = {{0.0f, 0.0f, 1.0f},
                                {0.8f, 0.4f, 1.0f},
                                {1.0f, 0.2f, 0.4f},
                                {0.6f, 0.0f, 1.0f},
                                {1.0f, 1.0f, 0.0f},
                                {0.0f, 1.0f, 1.0f}};

};

} //namespace ORB_SLAM

#endif // MAPDRAWER_H
