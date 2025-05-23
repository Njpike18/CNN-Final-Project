% Parameters (Adjusted for Rat Vision)
spatial_width = 15;
spatial_height = 15;
spatial_sigma_center = 1;
spatial_sigma_surround = 3;

% Create 2D Spatial Kernel (On-Center/Off-Surround)
[x, y] = meshgrid(-(spatial_width-1)/2:(spatial_width-1)/2, -(spatial_height-1)/2:(spatial_height-1)/2);
spatial_center = exp(-(x.^2 + y.^2) / (2 * spatial_sigma_center^2));
spatial_surround = exp(-(x.^2 + y.^2) / (2 * spatial_sigma_surround^2));
spatial_kernel = spatial_center - spatial_surround * (spatial_sigma_center / spatial_sigma_surround);
spatial_kernel = spatial_kernel / sum(abs(spatial_kernel(:))); % Normalize

% Input Pattern (2D Line at the first time point)
temporal_length = 7; % Assuming this is still the temporal length
input_line = zeros(100, 100, temporal_length);
input_line(50, :, 1) = 1; % Line along y-axis at the first time point

% Visualize the Spatial Kernel
figure;
subplot(1, 2, 1);
imagesc(spatial_kernel);
title('2D Spatial Receptive Field (Rat Model)');

colorbar;
axis square; % Make the aspect ratio square

% Visualize the Input Pattern (Line)
subplot(1, 2, 2);
imagesc(input_line(:, :, 1));
title('Input Pattern (Line)');

colorbar;
axis square;

disp('2D Spatial Receptive Field (spatial_kernel):');
disp(spatial_kernel);
disp(' ');
disp('Input Pattern (input_line at t=1):');
disp(input_line(:, :, 1));