# ROI轮廓/曲面特征提取工具使用说明

## 功能

当前第一版实现两个功能：

1. 二维ROI轮廓特征提取：输入灰度或二值掩膜图，输出面积、轮廓、质心、外接框、主轴方向、紧致度、轮廓曲率等特征。
2. 三维ROI曲面特征提取：输入XYZ点云，输出包围盒、质心、主方向、局部法向、曲率、粗糙度和边界候选点。

这些结果用于下一步候选拍照视点生成：

```text
ROI轮廓/曲面特征
-> 曲率自适应候选视点生成
-> Maskable PPO主动视点选择
```

## 文件结构

```text
src/trajectory_planning/roi_feature_extraction.py
scripts/extract_roi_features.py
scripts/demo_roi_feature_extraction.py
outputs/roi_feature_demo/
```

## 运行demo

```powershell
python .\scripts\demo_roi_feature_extraction.py
```

demo会生成：

```text
outputs/roi_feature_demo/demo_roi_mask.png
outputs/roi_feature_demo/demo_surface_cloud.csv
outputs/roi_feature_demo/demo_features.json
outputs/roi_feature_demo/demo_contour.csv
outputs/roi_feature_demo/demo_point_features.csv
```

## 提取二维ROI轮廓特征

```powershell
python .\scripts\extract_roi_features.py `
  --image-mask path\to\roi_mask.png `
  --output-json outputs\roi_features.json `
  --output-contour-csv outputs\roi_contour.csv
```

## 提取三维曲面特征

点云文件支持简单的TXT/CSV/XYZ格式，每行至少包含三列：

```text
x,y,z
```

运行：

```powershell
python .\scripts\extract_roi_features.py `
  --point-cloud path\to\surface.csv `
  --output-json outputs\surface_features.json `
  --output-point-features-csv outputs\surface_point_features.csv
```

## 从STL模型生成XYZ点云

如果已有转子叶片STL模型，可以先把STL表面采样成XYZ点云：

```powershell
python .\scripts\sample_stl_point_cloud.py `
  --stl path\to\blade.stl `
  --output outputs\blade_surface_points.csv `
  --points 20000
```

然后再对采样点云提取曲面特征：

```powershell
python .\scripts\extract_roi_features.py `
  --point-cloud outputs\blade_surface_points.csv `
  --output-json outputs\blade_surface_features.json `
  --output-point-features-csv outputs\blade_surface_point_features.csv
```

如果STL是整片叶片或整级转子模型，建议先在CAD/网格软件里裁剪出单片叶片或某个检查部位，再进行采样。这样提取出的曲率、法向和边界候选点更适合后续生成局部拍照视点。

## 同时提取二维和三维特征

```powershell
python .\scripts\extract_roi_features.py `
  --image-mask path\to\roi_mask.png `
  --point-cloud path\to\surface.csv `
  --output-json outputs\roi_surface_features.json `
  --output-contour-csv outputs\roi_contour.csv `
  --output-point-features-csv outputs\surface_point_features.csv
```

## 可视化特征提取结果

二维ROI可视化：

```powershell
python .\scripts\visualize_roi_features.py `
  --features-json outputs\roi_feature_demo\demo_features.json `
  --mask outputs\roi_feature_demo\demo_roi_mask.png `
  --contour-csv outputs\roi_feature_demo\demo_contour.csv `
  --output-dir outputs\roi_feature_demo\visualization
```

三维点云可视化：

```powershell
python .\scripts\visualize_roi_features.py `
  --features-json outputs\blade_surface_features.json `
  --point-features-csv outputs\blade_surface_point_features.csv `
  --output-dir outputs\blade_visualization
```

会生成：

```text
surface_projection_xy.png
surface_projection_xz.png
surface_projection_yz.png
surface_3d_curvature.png
feature_summary.txt
```

## 主要输出字段

二维ROI：

1. `area_px`：ROI面积。
2. `perimeter_px`：轮廓周长。
3. `centroid_xy`：ROI质心。
4. `bbox_xyxy`：外接框。
5. `orientation_deg`：PCA主轴方向。
6. `major_axis_px` / `minor_axis_px`：主轴和次轴长度。
7. `mean_curvature` / `max_curvature`：轮廓曲率统计。

三维ROI：

1. `bbox_min_xyz` / `bbox_max_xyz`：点云包围盒。
2. `centroid_xyz`：点云质心。
3. `principal_axes`：整体主方向。
4. `axis_lengths`：主方向尺度。
5. `mean_curvature` / `max_curvature`：局部PCA曲率统计。
6. `mean_roughness`：局部粗糙度。
7. `boundary_candidate_count`：边界候选点数量。

## 当前限制

1. 二维轮廓排序采用轻量级极角排序，适合单个紧凑ROI；复杂多孔区域后续需要升级为更严格的轮廓追踪。
2. 三维曲面特征基于K近邻PCA，不依赖Open3D；后续如果点云量较大，可接入Open3D或PCL提高效率。
3. 当前只完成特征提取，下一步需要实现基于曲率和视场约束的候选视点生成。
