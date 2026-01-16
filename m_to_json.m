% MATLAB脚本：将stereoParams转换为JSON格式
% 使用方法：
%   1. 在MATLAB中标定完成后，导出stereoParams到工作空间
%   2. 运行此脚本: >> matlab_to_json
%   3. 生成 stereo_calibration.json

% 加载标定参数（如果从文件加载）
% load('stereo_calibration.mat');

% 检查stereoParams是否存在
if ~exist('stereoParams', 'var')
    error('错误：找不到 stereoParams 变量。请先在 Stereo Camera Calibrator 中导出参数。');
end

fprintf('开始转换标定参数...\n');

% 创建数据结构
calib_data = struct();

% 左相机内参矩阵（3x3）
% MATLAB的IntrinsicMatrix是转置的，需要转回来
calib_data.camera_matrix_left = stereoParams.CameraParameters1.IntrinsicMatrix';

% 左相机畸变系数 [k1, k2, p1, p2, k3]
calib_data.dist_coeffs_left = [...
    stereoParams.CameraParameters1.RadialDistortion(1), ...
    stereoParams.CameraParameters1.RadialDistortion(2), ...
    stereoParams.CameraParameters1.TangentialDistortion(1), ...
    stereoParams.CameraParameters1.TangentialDistortion(2), ...
    0];  % k3通常为0

% 右相机内参矩阵（3x3）
calib_data.camera_matrix_right = stereoParams.CameraParameters2.IntrinsicMatrix';

% 右相机畸变系数
calib_data.dist_coeffs_right = [...
    stereoParams.CameraParameters2.RadialDistortion(1), ...
    stereoParams.CameraParameters2.RadialDistortion(2), ...
    stereoParams.CameraParameters2.TangentialDistortion(1), ...
    stereoParams.CameraParameters2.TangentialDistortion(2), ...
    0];

% 旋转矩阵（3x3）
calib_data.R = stereoParams.RotationOfCamera2';

% 平移向量（3x1），转换为米
% MATLAB的TranslationOfCamera2单位是毫米
calib_data.T = stereoParams.TranslationOfCamera2' / 1000;

% 图像尺寸 [height, width]
imageSize = stereoParams.CameraParameters1.ImageSize;
calib_data.image_size = imageSize;

% 计算立体校正参数
fprintf('计算立体校正参数...\n');

% 提取单目相机参数
K1 = stereoParams.CameraParameters1.IntrinsicMatrix';
D1_radial = stereoParams.CameraParameters1.RadialDistortion;
D1_tangential = stereoParams.CameraParameters1.TangentialDistortion;

K2 = stereoParams.CameraParameters2.IntrinsicMatrix';
D2_radial = stereoParams.CameraParameters2.RadialDistortion;
D2_tangential = stereoParams.CameraParameters2.TangentialDistortion;

R_stereo = stereoParams.RotationOfCamera2';
T_stereo = stereoParams.TranslationOfCamera2' / 1000;  % 转换为米

% OpenCV风格的立体校正
% 注意：MATLAB和OpenCV的坐标系可能略有不同
alpha = 0;  % 裁剪参数，0表示最大裁剪

% 计算校正变换
% 这里使用简化方法，因为MATLAB和OpenCV的接口不完全相同
[R1, R2, P1, P2] = rectify_stereo_cameras(K1, K2, R_stereo, T_stereo, imageSize, alpha);

% 计算重投影矩阵Q
Q = compute_Q_matrix(P1, P2);

calib_data.R1 = R1;
calib_data.R2 = R2;
calib_data.P1 = P1;
calib_data.P2 = P2;
calib_data.Q = Q;

% ROI（感兴趣区域）
% 简化处理，使用整个图像
calib_data.roi_left = [0, 0, imageSize(2), imageSize(1)];
calib_data.roi_right = [0, 0, imageSize(2), imageSize(1)];

% 重投影误差
calib_data.mean_reprojection_error = stereoParams.MeanReprojectionError;

% 时间戳
calib_data.timestamp = datestr(now, 'yyyy-mm-ddTHH:MM:SS');

% 转换为JSON
fprintf('转换为JSON格式...\n');

% 尝试使用jsonencode（MATLAB R2016b+）
try
    jsonStr = jsonencode(calib_data);
catch
    % 如果jsonencode不可用，使用savejson（JSONlab）
    try
        jsonStr = savejson('', calib_data);
    catch
        % 如果都不可用，手动构建
        fprintf('警告：无法使用jsonencode或savejson，将手动构建JSON\n');
        jsonStr = struct_to_json(calib_data);
    end
end

% 保存为JSON文件
fid = fopen('stereo_calibration.json', 'w');
if fid == -1
    error('无法创建文件 stereo_calibration.json');
end
fprintf(fid, '%s', jsonStr);
fclose(fid);

fprintf('\n✅ 标定参数已保存到 stereo_calibration.json\n');

% 显示关键参数
fprintf('\n关键参数:\n');
baseline = norm(stereoParams.TranslationOfCamera2);
fprintf('  基线距离: %.1f mm = %.4f m\n', baseline, baseline/1000);
fprintf('  重投影误差: %.4f 像素\n', stereoParams.MeanReprojectionError);
fprintf('  左相机焦距: %.2f\n', K1(1,1));
fprintf('  右相机焦距: %.2f\n', K2(1,1));
fprintf('  Q[3,2]: %.6f\n', Q(4,3));

fprintf('\n下一步:\n');
fprintf('  运行: python diagnose_calibration.py\n');


% ========== 辅助函数 ==========

function [R1, R2, P1, P2] = rectify_stereo_cameras(K1, K2, R, T, imageSize, alpha)
    % 简化的立体校正（类似OpenCV的stereoRectify）
    
    % Rodrigues旋转向量
    [U, ~, V] = svd(R);
    R_rect = U * V';
    
    % 校正旋转
    r = R_rect * [1; 0; 0];
    r = r / norm(r);
    
    % 构建校正矩阵
    e1 = T / norm(T);
    e2 = cross([0; 0; 1], e1);
    e2 = e2 / norm(e2);
    e3 = cross(e1, e2);
    
    R_rect_left = [e1'; e2'; e3'];
    R_rect_right = R_rect_left * R';
    
    R1 = R_rect_left;
    R2 = R_rect_right;
    
    % 新的投影矩阵
    P1 = K1 * [eye(3), zeros(3,1)];
    P2 = K2 * [R, T];
    
    % 调整以减少畸变
    cx_left = (imageSize(2) - 1) / 2;
    cy = (imageSize(1) - 1) / 2;
    cx_right = cx_left;
    
    fx = (K1(1,1) + K2(1,1)) / 2;
    fy = (K1(2,2) + K2(2,2)) / 2;
    
    % 重新构建P1和P2
    P1 = [fx, 0, cx_left, 0;
          0, fy, cy, 0;
          0, 0, 1, 0];
    
    baseline = norm(T);
    P2 = [fx, 0, cx_right, -fx*baseline;
          0, fy, cy, 0;
          0, 0, 1, 0];
end

function Q = compute_Q_matrix(P1, P2)
    % 计算重投影矩阵Q
    
    cx = P1(1,3);
    cy = P1(2,3);
    fx = P1(1,1);
    
    % 基线距离
    Tx = -P2(1,4) / P2(1,1);
    
    Q = [1, 0, 0, -cx;
         0, 1, 0, -cy;
         0, 0, 0, fx;
         0, 0, -1/Tx, 0];
end

function jsonStr = struct_to_json(s)
    % 简单的struct到JSON转换（手动实现）
    % 仅用于备用
    
    jsonStr = '{';
    fields = fieldnames(s);
    
    for i = 1:length(fields)
        field = fields{i};
        value = s.(field);
        
        jsonStr = [jsonStr, '"', field, '":'];
        
        if isnumeric(value)
            if numel(value) == 1
                jsonStr = [jsonStr, num2str(value, '%.6f')];
            else
                jsonStr = [jsonStr, matrix_to_json(value)];
            end
        elseif ischar(value)
            jsonStr = [jsonStr, '"', value, '"'];
        end
        
        if i < length(fields)
            jsonStr = [jsonStr, ','];
        end
    end
    
    jsonStr = [jsonStr, '}'];
end

function jsonStr = matrix_to_json(mat)
    % 矩阵到JSON数组
    [rows, cols] = size(mat);
    
    jsonStr = '[';
    for i = 1:rows
        jsonStr = [jsonStr, '['];
        for j = 1:cols
            jsonStr = [jsonStr, num2str(mat(i,j), '%.6f')];
            if j < cols
                jsonStr = [jsonStr, ','];
            end
        end
        jsonStr = [jsonStr, ']'];
        if i < rows
            jsonStr = [jsonStr, ','];
        end
    end
    jsonStr = [jsonStr, ']'];
end
