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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <algorithm>
#include <mutex>

#include <open3d/Open3D.h>
#include "Optimizer.h"

namespace ORB_SLAM3
{


MapDrawer::MapDrawer(Atlas* pAtlas, const string &strSettingPath, Settings* settings):mpAtlas(pAtlas)
{
    if(settings){
        newParameterLoader(settings);
    }
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        bool is_correct = ParseViewerParamFile(fSettings);

        if(!is_correct)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }
}

void MapDrawer::newParameterLoader(Settings *settings) {
    mKeyFrameSize = settings->keyFrameSize();
    mKeyFrameLineWidth = settings->keyFrameLineWidth();
    mGraphLineWidth = settings->graphLineWidth();
    mPointSize = settings->pointSize();
    mCameraSize = settings->cameraSize();
    mCameraLineWidth  = settings->cameraLineWidth();
}

bool MapDrawer::ParseViewerParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::FileNode node = fSettings["Viewer.KeyFrameSize"];
    if(!node.empty())
    {
        mKeyFrameSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.KeyFrameLineWidth"];
    if(!node.empty())
    {
        mKeyFrameLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.GraphLineWidth"];
    if(!node.empty())
    {
        mGraphLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.GraphLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.PointSize"];
    if(!node.empty())
    {
        mPointSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.PointSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraSize"];
    if(!node.empty())
    {
        mCameraSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraLineWidth"];
    if(!node.empty())
    {
        mCameraLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}

void MapDrawer::DrawMapPoints()
{
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if(!pActiveMap)
        return;

    const vector<MapPoint*> &vpMPs = pActiveMap->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = pActiveMap->GetReferenceMapPoints();

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        if(vpMPs[i]->HasColor())
        {
            const cv::Vec3b color = vpMPs[i]->GetColor();
            glColor3f(color[2] / 255.f, color[1] / 255.f, color[0] / 255.f);
        }
        else
        {
            glColor3f(0.2f, 0.2f, 0.2f);
        }
        Eigen::Matrix<float,3,1> pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;
        if((*sit)->HasColor())
        {
            const cv::Vec3b color = (*sit)->GetColor();
            glColor3f(color[2] / 255.f, color[1] / 255.f, color[0] / 255.f);
        }
        else
        {
            glColor3f(1.0f, 0.0f, 0.0f);
        }
        Eigen::Matrix<float,3,1> pos = (*sit)->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));

    }

    glEnd();
}

void MapDrawer::DrawVGGTDenseCloud(bool onlyActiveMap, size_t maxPoints, float pointSizeOverride)
{
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if(!pActiveMap || maxPoints == 0)
        return;

    std::vector<Map*> maps = mpAtlas->GetAllMaps();
    if(onlyActiveMap)
    {
        maps.clear();
        maps.push_back(pActiveMap);
    }

    // First pass to estimate density for decimation on the newest KF of each map.
    size_t total_points = 0;
    std::vector<KeyFrame*> latest_kfs;
    latest_kfs.reserve(maps.size());
    for(Map* pMap : maps)
    {
        if(!pMap)
            continue;
        const std::vector<KeyFrame*> kfs = pMap->GetAllKeyFrames();
        KeyFrame* latest = nullptr;
        for(KeyFrame* kf : kfs)
        {
            if(!kf)
                continue;
            if(!latest || kf->mnId > latest->mnId)
                latest = kf;
        }
        if(latest)
        {
            latest_kfs.push_back(latest);
            total_points += latest->GetVGGTDenseMapPoints().size();
        }
    }

    if(total_points == 0)
        return;

    const size_t stride = std::max<size_t>(1, total_points / maxPoints);
    const float point_size = pointSizeOverride > 0.0f ? pointSizeOverride : mPointSize;

    glPointSize(point_size);
    glBegin(GL_POINTS);

    size_t emitted = 0;
    size_t idx = 0;
    for(KeyFrame* kf : latest_kfs)
    {
        if(!kf)
            continue;
        const std::vector<VGGTDensePointRGBXYZ> dense = kf->GetVGGTDenseMapPoints();
        for(const auto& pt : dense)
        {
            if((idx++ % stride) != 0)
                continue;
            if(emitted >= maxPoints)
                break;

            glColor3f(pt.rgb[2] / 255.f, pt.rgb[1] / 255.f, pt.rgb[0] / 255.f);
            glVertex3f(pt.xyz.x(), pt.xyz.y(), pt.xyz.z());
            ++emitted;
        }
        if(emitted >= maxPoints)
            break;
    }

    glEnd();
}

void MapDrawer::DrawTSDFMesh(bool wireframe, size_t maxFaces, float lineWidth, float faceAlpha)
{
    if(maxFaces == 0)
    {
        return;
    }

    auto mesh_ptr = ExtractTSDFMeshCopy();
    if(!mesh_ptr)
    {
        return;
    }

    auto& mesh = *mesh_ptr;
    if(mesh.triangles_.empty() || mesh.vertices_.empty())
    {
        return;
    }

    const size_t tri_count = mesh.triangles_.size();
    const size_t stride = std::max<size_t>(1, tri_count / maxFaces);

    if(wireframe)
    {
        glLineWidth(lineWidth > 0.0f ? lineWidth : mCameraLineWidth);
        glBegin(GL_LINES);
    }
    else
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBegin(GL_TRIANGLES);
    }

    const bool has_color = mesh.HasVertexColors();
    size_t emitted = 0;
    for(size_t i = 0; i < tri_count; ++i)
    {
        if((i % stride) != 0)
            continue;
        if(emitted >= maxFaces)
            break;

        const Eigen::Vector3i &tri = mesh.triangles_[i];
        const Eigen::Vector3d &v0 = mesh.vertices_[tri(0)];
        const Eigen::Vector3d &v1 = mesh.vertices_[tri(1)];
        const Eigen::Vector3d &v2 = mesh.vertices_[tri(2)];

        auto set_color = [&](int idx)
        {
            if(has_color && idx < static_cast<int>(mesh.vertex_colors_.size()))
            {
                const Eigen::Vector3d &c = mesh.vertex_colors_[idx];
                glColor4f(static_cast<float>(c(0)), static_cast<float>(c(1)), static_cast<float>(c(2)), faceAlpha);
            }
            else
            {
                glColor4f(0.2f, 0.6f, 1.0f, faceAlpha);
            }
        };

        if(wireframe)
        {
            set_color(tri(0));
            glVertex3d(v0(0), v0(1), v0(2));
            glVertex3d(v1(0), v1(1), v1(2));

            set_color(tri(1));
            glVertex3d(v1(0), v1(1), v1(2));
            glVertex3d(v2(0), v2(1), v2(2));

            set_color(tri(2));
            glVertex3d(v2(0), v2(1), v2(2));
            glVertex3d(v0(0), v0(1), v0(2));
        }
        else
        {
            set_color(tri(0));
            glVertex3d(v0(0), v0(1), v0(2));
            set_color(tri(1));
            glVertex3d(v1(0), v1(1), v1(2));
            set_color(tri(2));
            glVertex3d(v2(0), v2(1), v2(2));
        }

        ++emitted;
    }

    glEnd();

    if(!wireframe)
    {
        glDisable(GL_BLEND);
    }
}

void MapDrawer::AddFramePose(const Sophus::SE3f &Tcw, bool isKeyFrame)
{
    // Store Twc (world pose) for drawing; limit size to avoid unbounded growth
    std::lock_guard<std::mutex> lock(mMutexTrajectory);
    const size_t kMaxStored = 20000; // configurable cap
    mFrameTrajectory.push_back(Tcw.inverse());
    mFrameIsKeyFrame.push_back(isKeyFrame);
    if(mFrameTrajectory.size() > kMaxStored)
    {
        // Drop oldest block (simple strategy)
        mFrameTrajectory.erase(mFrameTrajectory.begin(), mFrameTrajectory.begin()+ (mFrameTrajectory.size()/10));
        mFrameIsKeyFrame.erase(mFrameIsKeyFrame.begin(), mFrameIsKeyFrame.begin()+ (mFrameIsKeyFrame.size()/10));
    }
}

void MapDrawer::ClearFrameTrajectory()
{
    std::lock_guard<std::mutex> lock(mMutexTrajectory);
    mFrameTrajectory.clear();
    mFrameIsKeyFrame.clear();
}

void MapDrawer::DrawFrameTrajectory()
{
    std::lock_guard<std::mutex> lock(mMutexTrajectory);
    if(mFrameTrajectory.empty()) return;

    // Draw as small points and a polyline: non-keyframes in light gray, keyframes (if stored) in cyan.
    glPointSize(std::max(1.0f, mPointSize * 0.5f));
    glBegin(GL_POINTS);
    for(size_t i=0; i<mFrameTrajectory.size(); ++i)
    {
        const Sophus::SE3f &Twc = mFrameTrajectory[i];
        Eigen::Vector3f t = Twc.translation();
        if(mFrameIsKeyFrame[i])
            glColor3f(0.0f, 0.8f, 0.8f); // keyframe color (different from KF boxes)
        else
            glColor3f(0.5f, 0.5f, 0.5f); // non-keyframe
        glVertex3f(t.x(), t.y(), t.z());
    }
    glEnd();

    // Polyline of trajectory
    glLineWidth(1.0f);
    glBegin(GL_LINE_STRIP);
    glColor3f(0.3f,0.3f,0.3f);
    for(size_t i=0; i<mFrameTrajectory.size(); ++i)
    {
        const Sophus::SE3f &Twc = mFrameTrajectory[i];
        Eigen::Vector3f t = Twc.translation();
        glVertex3f(t.x(), t.y(), t.z());
    }
    glEnd();
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph, const bool bDrawOptLba)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    Map* pActiveMap = mpAtlas->GetCurrentMap();
    // DEBUG LBA
    std::set<long unsigned int> sOptKFs = pActiveMap->msOptKFs;
    std::set<long unsigned int> sFixedKFs = pActiveMap->msFixedKFs;

    if(!pActiveMap)
        return;

    const vector<KeyFrame*> vpKFs = pActiveMap->GetAllKeyFrames();

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
            unsigned int index_color = pKF->mnOriginMapId;
            (void)index_color;

            glPushMatrix();

            glMultMatrixf((GLfloat*)Twc.data());

            if(!pKF->GetParent()) // It is the first KF in the map
            {
                glLineWidth(mKeyFrameLineWidth*5);
                glColor3f(1.0f,0.0f,0.0f);
                glBegin(GL_LINES);
            }
            else
            {
                //cout << "Child KF: " << vpKFs[i]->mnId << endl;
                glLineWidth(mKeyFrameLineWidth);
                if (bDrawOptLba) {
                    if(sOptKFs.find(pKF->mnId) != sOptKFs.end())
                    {
                        glColor3f(0.0f,1.0f,0.0f); // Green -> Opt KFs
                    }
                    else if(sFixedKFs.find(pKF->mnId) != sFixedKFs.end())
                    {
                        glColor3f(1.0f,0.0f,0.0f); // Red -> Fixed KFs
                    }
                    else
                    {
                        glColor3f(0.0f,0.0f,1.0f); // Basic color
                    }
                }
                else
                {
                    glColor3f(0.0f,0.0f,1.0f); // Basic color
                }
                glBegin(GL_LINES);
            }

            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();

            glEnd();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        // cout << "-----------------Draw graph-----------------" << endl;
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            Eigen::Vector3f Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    Eigen::Vector3f Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow(0),Ow(1),Ow(2));
                    glVertex3f(Ow2(0),Ow2(1),Ow2(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                Eigen::Vector3f Owp = pParent->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                Eigen::Vector3f Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owl(0),Owl(1),Owl(2));
            }
        }

        glEnd();
    }

    if(bDrawInertialGraph && pActiveMap->isImuInitialized())
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(1.0f,0.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        //Draw inertial links
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            Eigen::Vector3f Ow = pKFi->GetCameraCenter();
            KeyFrame* pNext = pKFi->mNextKF;
            if(pNext)
            {
                Eigen::Vector3f Owp = pNext->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }
        }

        glEnd();
    }

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();

    if(bDrawKF)
    {
        for(Map* pMap : vpMaps)
        {
            if(pMap == pActiveMap)
                continue;

            vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

            for(size_t i=0; i<vpKFs.size(); i++)
            {
                KeyFrame* pKF = vpKFs[i];
                Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
                unsigned int index_color = pKF->mnOriginMapId;

                glPushMatrix();

                glMultMatrixf((GLfloat*)Twc.data());

                if(!vpKFs[i]->GetParent()) // It is the first KF in the map
                {
                    glLineWidth(mKeyFrameLineWidth*5);
                    glColor3f(1.0f,0.0f,0.0f);
                    glBegin(GL_LINES);
                }
                else
                {
                    glLineWidth(mKeyFrameLineWidth);
                    glColor3f(mfFrameColors[index_color][0],mfFrameColors[index_color][1],mfFrameColors[index_color][2]);
                    glBegin(GL_LINES);
                }

                glVertex3f(0,0,0);
                glVertex3f(w,h,z);
                glVertex3f(0,0,0);
                glVertex3f(w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,h,z);

                glVertex3f(w,h,z);
                glVertex3f(w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(-w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(w,h,z);

                glVertex3f(-w,-h,z);
                glVertex3f(w,-h,z);
                glEnd();

                glPopMatrix();
            }
        }
    }
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}


void MapDrawer::SetCurrentCameraPose(const Sophus::SE3f &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.inverse();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M, pangolin::OpenGlMatrix &MOw)
{
    Eigen::Matrix4f Twc;
    {
        unique_lock<mutex> lock(mMutexCamera);
        Twc = mCameraPose.matrix();
    }

    for (int i = 0; i<4; i++) {
        M.m[4*i] = Twc(0,i);
        M.m[4*i+1] = Twc(1,i);
        M.m[4*i+2] = Twc(2,i);
        M.m[4*i+3] = Twc(3,i);
    }

    MOw.SetIdentity();
    MOw.m[12] = Twc(0,3);
    MOw.m[13] = Twc(1,3);
    MOw.m[14] = Twc(2,3);
}
} //namespace ORB_SLAM
