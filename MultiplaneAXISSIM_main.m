% =========================================================================
% Multiplane AXIS-SIM : reconstruction + comparison display
% RMS / RMS_decon8 / SACD / 3D SOFI  ->  XY MIP + XZ cross section
% -------------------------------------------------------------------------
% *** PLEASE CITE ALL THREE ***
%   1) Multiplane AXIS-SIM (this work)
%   2) AXIS-SIM            https://doi.org/10.1038/s41467-025-64366-2
%   3) SOFI package        https://github.com/kgrussmayer/sofipackage  (GPL)
% A citation costs one line and keeps these tools open. Thank you.
% =========================================================================
clc; clear; close all;
t0 = tic;
stage = @(s) fprintf('[%6.1f s] %s\n', toc(t0), s);

%% Paths
baseDir = fileparts(mfilename('fullpath'));
if isempty(baseDir); baseDir = pwd; end
dataDir = fullfile(baseDir, 'data');
resultsDir = fullfile(baseDir, 'results');
for d = {'functions','utils'}
    p = fullfile(baseDir, d{1});
    if exist(p,'dir'); addpath(genpath(p)); end
end
if ~exist(resultsDir,'dir'); mkdir(resultsDir); end

needed = {'sofiCumulants3D','sofiGrids','statusbar','cumulant', ...
          'fourierInterpolation','convnfft','cpu'};
miss = needed(cellfun(@(f) isempty(which(f)), needed));
if ~isempty(miss); error('Missing on path: %s', strjoin(miss,', ')); end

%% Parameters
layer_num = 100;  total_layer = 6;  crop = 10;
scale_factor = 30/35;  selected_planes = [2 3 5 6];  transition_frame = 100;
ref_planes = 1:3;  target_planes = 4:6;
iter_pre = 7;  iter_post = 8;  ACorder = 3;  mag = 3;
subfactor = 0.8;  bg_sigma = 300;  bg_floor = 0.01;  use_gpu = false;

% display
voxRMS  = [135.4 135.4 1029];      % [dy dx dz] nm
voxSACD = [45.1  45.1  1029];
voxSOFI = [45.1  45.1  343];
targetVox   = 45.1;                % isotropic display voxel [nm]
sigmaZ      = [0.8 0.8 0.8 1.0];   % z low-pass per volume
sigmaXY     = 0.6;                 % 0 = off
displayPrct = [0.5 99.8];
barXY_nm    = 2000;  barXZ_nm = 1000;
p1_ref = [704 574];                % else use these (reference = SOFI MIP px)
p2_ref = [812 666];

norm3d = @(x) (x - min(x(:))) ./ (max(x(:)) - min(x(:)));
to16   = @(x) double(uint16(norm3d(x)*65535));

%% Load
stage('load PSFs');
psf_3DBW   = readTiffStack(fullfile(dataDir,'PSF BW_r1.333_NA100_w515_xy135.4_z1029.tif'));
psf_3DBWv2 = readTiffStack(fullfile(dataDir,'PSF BW_r1.333_NA100_w515_xy45.1_z1029.tif'));
psf_3DBWv3 = readTiffStack(fullfile(dataDir,'PSF BW_r1.333_NA100_w515_xy45.1_z343.tif'));
psf = psf_3DBW(:,:,65);  psfv2 = psf_3DBWv2(:,:,65);  psfv3 = psf_3DBWv3(:,:,65);

stage('load planes');
planeFile = @(p) fullfile(dataDir, sprintf('alignmented_plane%d.tif', p));
info = imfinfo(planeFile(1));
A_layer = zeros(info(1).Height, info(1).Width, layer_num, total_layer);
for p = 1:total_layer
    for f = 1:layer_num
        A_layer(:,:,f,p) = double(imread(planeFile(p), f));
    end
end
A_layer = A_layer(1+crop:end-crop, 1+crop:end-crop, :, :);
[ny, nx, ~, ~] = size(A_layer);

stage('normalize planes');
A_layer(:,:,1:transition_frame,selected_planes) = ...
    A_layer(:,:,1:transition_frame,selected_planes) * scale_factor;
sf2 = mean(A_layer(:,:,:,ref_planes),'all') / mean(A_layer(:,:,:,target_planes),'all');
A_layer(:,:,:,target_planes) = A_layer(:,:,:,target_planes) * sf2;

%% RMS
stage('RMS');
RMS = squeeze(std(A_layer, 1, 3));
RMS_decon = deconvlucy(to16(RMS), psf_3DBW(:,:,63:67), iter_post);
writeTiffStack(uint16(norm3d(RMS)*65535), fullfile(resultsDir, sprintf('cos7_488_RMS.tif')));
writeTiffStack(uint16(norm3d(RMS_decon)*65535), fullfile(resultsDir, sprintf('cos7_488_RMS_decon8.tif')));
stage('save RMS');

%% Pre-deconvolution + DSI
stage('pre-deconvolution');
datadeconDSI3D = zeros(ny, nx, total_layer, layer_num);
for p = 1:total_layer
    dd = zeros(ny, nx, layer_num);
    for f = 1:layer_num
        dd(:,:,f) = deconvlucy(A_layer(:,:,f,p), psf, iter_pre);
    end
    datadeconDSI3D(:,:,p,:) = permute(std(dd,1,3) .* dd, [1 2 4 3]);
    stage(sprintf('  plane %d/%d', p, total_layer));
end
datadeconDSI3D_linear = sqrt(datadeconDSI3D);

%% SACD
stage('SACD');
cumDSI = zeros(ny*mag, nx*mag, total_layer);
for p = 1:total_layer
    tmp = fourierInterpolation(squeeze(datadeconDSI3D_linear(:,:,p,:)), [mag mag 1], 'lateral');
    tmp(tmp < 0) = 0;
    tmp = abs(tmp - subfactor*mean(tmp,3));
    cumDSI(:,:,p) = abs(cumulant(tmp, ACorder));
end
SACD = deconvlucy(cumDSI, psf_3DBWv2(:,:,63:67).^ACorder, iter_post);

%% 3D SOFI
stage('3D SOFI');
sofi3D = sofiCumulants3D(datadeconDSI3D_linear, [], [], [], ACorder, use_gpu);
imsofi3d = sofi3D{ACorder};
imsofi3d(imsofi3d < 0) = 0;
imsofi3d = deconvlucy(imsofi3d, psf_3DBWv3(:,:,58:73).^ACorder, iter_post);

%% Finalize + save
stage('finalize');
SACD_out = finalize(to16(SACD), psf_3DBWv2(:,:,63:67), psfv2, iter_post, ACorder, bg_sigma, bg_floor);
sofi_out = finalize(to16(imsofi3d), psf_3DBWv3(:,:,58:73), psfv3, iter_post, ACorder, bg_sigma, bg_floor);
writeTiffStack(uint16(norm3d(SACD_out)*65535), fullfile(resultsDir, sprintf('cos7_488_SACDDSI3d_bg300.tif')));
writeTiffStack(uint16(norm3d(sofi_out)*65535), fullfile(resultsDir, sprintf('cos7_488_sofi3dDSI3d_bg300.tif')));
stage('saved');

%% ===================== COMPARISON DISPLAY ===============================
stage('display');
names = {'RMS','RMS decon8','SACD','3D SOFI'};
vols  = {double(RMS), double(RMS_decon), double(SACD_out), double(sofi_out)};
voxs  = {voxRMS, voxRMS, voxSACD, voxSOFI};
 
disp_vols = cell(1,4);
for i = 1:4
    v = smoothAlongZ(vols{i}, sigmaZ(i));
    if sigmaXY > 0; v = smoothXY(v, sigmaXY); end
    disp_vols{i} = resampleIso(v, voxs{i}, targetVox);
end
 
% common black canvas: pad every volume to the largest size, centered
S = cell2mat(cellfun(@(v) size(v,1:3), disp_vols(:), 'UniformOutput', false));
canvas = max(S, [], 1);
for i = 1:4
    disp_vols{i} = padCenter(disp_vols{i}, canvas);
end
 
mipXY = cellfun(@(v) max(v,[],3), disp_vols, 'UniformOutput', false);
refSize = size(mipXY{4});
xz = cellfun(@(v) extractXZ(v, p1_ref, p2_ref, refSize), disp_vols, 'UniformOutput', false);
 
barXY_px = round(barXY_nm/targetVox);
barXZ_px = round(barXZ_nm/targetVox);
 
figure('Color','k','Name','RMS / decon8 / SACD / SOFI', ...
       'Position',[60 60 1600 800],'InvertHardcopy','off');
for i = 1:4
    subplot(2,4,i);
    showRobust(mipXY{i}, displayPrct); axis image off; hold on;
    plot([p1_ref(1) p2_ref(1)], [p1_ref(2) p2_ref(2)], 'c-', 'LineWidth', 1.2);
    addBar(size(mipXY{i}), barXY_px, max(4,round(barXY_px/6)), barXY_nm);
    hold off; title([names{i} ' - XY MIP'], 'Color','w');
 
    subplot(2,4,4+i);
    showRobust(xz{i}, displayPrct); axis image; set(gca,'YDir','normal');
    set(gca,'XTick',[],'YTick',[],'XColor','w','YColor','w'); hold on;
    addBar(size(xz{i}), barXZ_px, 4, barXZ_nm);
    hold off; title([names{i} ' - XZ section'], 'Color','w');
end
colormap hot;
sgtitle(sprintf('isotropic %.1f nm', ...
    targetVox), 'FontWeight','bold','Color','w');
stage('done');

%% ============================ FUNCTIONS =================================
function out = finalize(vol, psf3d, psf2d, iter, order, bg_sigma, bg_floor)
vol = deconvlucy(vol, psf3d, iter);
vol = vol.^(1/order);
for z = 1:size(vol,3)
    vol(:,:,z) = vol(:,:,z) - imgaussfilt(vol(:,:,z), bg_sigma);
end
vol(vol < -bg_floor) = -bg_floor;  vol = vol + bg_floor;
out = zeros(size(vol));
for z = 1:size(vol,3)
    out(:,:,z) = convnfft(vol(:,:,z), psf2d.^order, 'same');
end
end

function stack = readTiffStack(fname)
info = imfinfo(fname);
stack = zeros(info(1).Height, info(1).Width, numel(info));
for i = 1:numel(info); stack(:,:,i) = double(imread(fname, i)); end
end

function writeTiffStack(stack, fname)
if exist(fname,'file'); delete(fname); end
for i = 1:size(stack,3); imwrite(stack(:,:,i), fname, 'WriteMode','append'); end
end

function v = smoothAlongZ(v, s)
if s <= 0 || size(v,3) < 3; return; end
r = max(1, ceil(3*s));  g = exp(-((-r:r).^2)/(2*s^2));  g = g/sum(g);
v = convn(v, reshape(g,1,1,[]), 'same');
end

function v = smoothXY(v, s)
if s <= 0; return; end
for k = 1:size(v,3); v(:,:,k) = imgaussfilt(v(:,:,k), s); end
end

function vo = resampleIso(v, vox, target)
sc = [vox(1) vox(2) vox(3)]/target;
vo = imresize3(v, max(round(size(v).*sc),1), 'nearest');
end

function showRobust(img, prct)
lo = prctile(img(:), prct(1));  hi = prctile(img(:), prct(2));
if hi <= lo; lo = min(img(:)); hi = max(img(:)); end
imagesc(img, [lo hi]); axis image;
end

function addBar(sz, len, thick, len_nm)
x0 = sz(2) - len - 10;  y0 = sz(1) - thick - 19;
rectangle('Position', [x0, y0, len, thick], 'EdgeColor','w', 'FaceColor','w');
text(x0 + len/2, y0 - 10, sprintf('%g \\mum', len_nm/1000), ...
     'Color','w', 'HorizontalAlignment','center', ...
     'VerticalAlignment','bottom', 'FontWeight','bold', 'FontSize', 9);
end

function xzImg = extractXZ(vol, p1, p2, refSize)
[H, W, Z] = size(vol);
x1 = (p1(1)-1)/max(refSize(2)-1,1)*max(W-1,1) + 1;
y1 = (p1(2)-1)/max(refSize(1)-1,1)*max(H-1,1) + 1;
x2 = (p2(1)-1)/max(refSize(2)-1,1)*max(W-1,1) + 1;
y2 = (p2(2)-1)/max(refSize(1)-1,1)*max(H-1,1) + 1;
L  = round(hypot(x2-x1, y2-y1)) + 1;
xs = linspace(x1,x2,L);  ys = linspace(y1,y2,L);
xzImg = zeros(Z, L);
for k = 1:Z
    xzImg(k,:) = improfile(vol(:,:,k), xs, ys, L, 'bilinear');
end
xzImg = fillmissing(xzImg, 'nearest');
end

function vo = padCenter(v, canvas)
vo = zeros(canvas, 'like', v);
s  = size(v, 1:3);
o  = floor((canvas - s)/2);
vo(o(1)+(1:s(1)), o(2)+(1:s(2)), o(3)+(1:s(3))) = v;
end
