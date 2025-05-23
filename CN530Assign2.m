% 1D Spatio-Temporal Edge Detection using On-Center/Off-Surround Kernels

% Clear workspace and close figures to ensure a clean slate.
clear;
close all;

% Parameters
spatial_width = 21; % Spatial extent
temporal_length = 5; % Temporal extent
spatial_sigma_center = 2; % Center Gaussian sigma (spatial)
spatial_sigma_surround = 5; % Surround Gaussian sigma (spatial)
temporal_sigma = 1; % Temporal Gaussian sigma

% Create Spatial Kernels (On-Center/Off-Surround)
spatial_center = exp(-(linspace(-(spatial_width-1)/2, (spatial_width-1)/2, spatial_width).^2) / (2 * spatial_sigma_center^2));
spatial_surround = exp(-(linspace(-(spatial_width-1)/2, (spatial_width-1)/2, spatial_width).^2) / (2 * spatial_sigma_surround^2));
spatial_kernel = spatial_center - (spatial_surround * (spatial_sigma_center/spatial_sigma_surround)); % Difference of Gaussians

% Normalize the spatial kernel to sum to zero (approx.)
spatial_kernel = spatial_kernel / sum(abs(spatial_kernel));

% Create Temporal Kernel (Gaussian)
temporal_kernel = exp(-(linspace(0, temporal_length-1, temporal_length).^2) / (2 * temporal_sigma^2));
temporal_kernel = temporal_kernel / sum(temporal_kernel); % Normalize

% Create 1D Spatio-Temporal Kernel (Separable)
spatiotemporal_kernel = zeros(spatial_width, temporal_length);
for t = 1:temporal_length
    spatiotemporal_kernel(:, t) = spatial_kernel * temporal_kernel(t);
end

% Create Input Signal (Moving Edge)
input_signal = zeros(100, temporal_length);
edge_start = 20;
edge_speed = 2;

for t = 1:temporal_length
    edge_location = edge_start + (t - 1) * edge_speed;
    input_signal(1:edge_location, t) = 1;
end

% Convolution (Spatio-Temporal)
output_signal = zeros(size(input_signal)); % Preallocate

for t = 1:temporal_length
    input_padded = [zeros((spatial_width-1)/2, 1); input_signal(:, t); zeros((spatial_width-1)/2, 1)];

    for x = 1:size(input_signal, 1)
        convolution_result = sum(input_padded(x:x+spatial_width-1) .* spatial_kernel);
        output_signal(x, t) = convolution_result(1); % Force scalar assignment
    end
end

% Temporal Filtering
final_output = zeros(size(input_signal));

for x = 1:size(input_signal, 1)
    temp_output = conv(output_signal(x, :), temporal_kernel, 'same');
    final_output(x,:) = temp_output;
end

% Plotting
figure;
subplot(3, 1, 1);
imagesc(input_signal');
title('Input Signal (Moving Edge)');
xlabel('Spatial Position');
ylabel('Time');

subplot(3, 1, 2);
imagesc(output_signal');
title('Spatial Convolution Output');
xlabel('Spatial Position');
ylabel('Time');

subplot(3, 1, 3);
imagesc(final_output');
title('Spatio-Temporal Output');
xlabel('Spatial Position');
ylabel('Time');

% Equilibrium State Graph (Steady State)
equilibrium_input = zeros(100,1);
equilibrium_input(1:50) = 1;

equilibrium_output = zeros(size(equilibrium_input));
equilibrium_input_padded = [zeros((spatial_width-1)/2, 1); equilibrium_input; zeros((spatial_width-1)/2, 1)];

for x = 1:size(equilibrium_input, 1)
    convolution_result = sum(input_padded(x:x+spatial_width-1) .* spatial_kernel);
    equilibrium_output(x) = convolution_result(1); % Force scalar assignment
end

figure;
subplot(2,1,1);
plot(equilibrium_input);
title('Equilibrium Input');
xlabel('Spatial Position');
ylabel('Intensity');
subplot(2,1,2);
plot(equilibrium_output);
title('Equilibrium Output');
xlabel('Spatial Position');
ylabel('Response');



%%



% 1D Spatio-Temporal Edge Detection with Line/Surface Inputs

% Clear workspace and close figures.
clear;
close all;

% Parameters (Adjustable)
spatial_width = 21; % Spatial extent of kernel
temporal_length = 5; % Temporal extent
spatial_sigma_center = 2; % Center Gaussian sigma (spatial)
spatial_sigma_surround = 5; % Surround Gaussian sigma (spatial)
temporal_sigma = 1; % Temporal Gaussian sigma

% Create Spatial Kernels (On-Center/Off-Surround)
spatial_center = exp(-(linspace(-(spatial_width-1)/2, (spatial_width-1)/2, spatial_width).^2) / (2 * spatial_sigma_center^2));
spatial_surround = exp(-(linspace(-(spatial_width-1)/2, (spatial_width-1)/2, spatial_width).^2) / (2 * spatial_sigma_surround^2));
spatial_kernel = spatial_center - (spatial_surround * (spatial_sigma_center/spatial_sigma_surround));
spatial_kernel = spatial_kernel / sum(abs(spatial_kernel)); % Normalize

% Create Temporal Kernel (Gaussian)
temporal_kernel = exp(-(linspace(0, temporal_length-1, temporal_length).^2) / (2 * temporal_sigma^2));
temporal_kernel = temporal_kernel / sum(temporal_kernel);

% Create Spatio-Temporal Kernel
spatiotemporal_kernel = zeros(spatial_width, temporal_length);
for t = 1:temporal_length
    spatiotemporal_kernel(:, t) = spatial_kernel * temporal_kernel(t);
end

% Input Signals (Line, Thin Surface, Thick Surface)
input_line = zeros(100, temporal_length);
input_line(50, :) = 1; % Line at center

input_thin_surface = zeros(100, temporal_length);
input_thin_surface(48:52, :) = 1; % Thin surface

input_thick_surface = zeros(100, temporal_length);
input_thick_surface(40:60, :) = 1; % Thick surface

% Apply the Filter to Each Input
output_line = spatio_temporal_filter(input_line, spatial_kernel, temporal_kernel);
output_thin_surface = spatio_temporal_filter(input_thin_surface, spatial_kernel, temporal_kernel);
output_thick_surface = spatio_temporal_filter(input_thick_surface, spatial_kernel, temporal_kernel);

% Plotting Spatio-Temporal Responses
figure;
subplot(3, 3, 1); imagesc(input_line'); title('Input Line');
subplot(3, 3, 2); imagesc(output_line'); title('Output Line');
subplot(3, 3, 4); imagesc(input_thin_surface'); title('Input Thin Surface');
subplot(3, 3, 5); imagesc(output_thin_surface'); title('Output Thin Surface');
subplot(3, 3, 7); imagesc(input_thick_surface'); title('Input Thick Surface');
subplot(3, 3, 8); imagesc(output_thick_surface'); title('Output Thick Surface');

% Equilibrium State (Static Input)
equilibrium_line = zeros(100, 1); equilibrium_line(50) = 1;
equilibrium_thin_surface = zeros(100, 1); equilibrium_thin_surface(48:52) = 1;
equilibrium_thick_surface = zeros(100, 1); equilibrium_thick_surface(40:60) = 1;

equilibrium_output_line = conv(equilibrium_line, spatial_kernel, 'same');
equilibrium_output_thin_surface = conv(equilibrium_thin_surface, spatial_kernel, 'same');
equilibrium_output_thick_surface = conv(equilibrium_thick_surface, spatial_kernel, 'same');

% Plotting Equilibrium Responses
figure;
subplot(3, 2, 1); plot(equilibrium_line); title('Equilibrium Line Input');
subplot(3, 2, 2); plot(equilibrium_output_line); title('Equilibrium Line Output');
subplot(3, 2, 3); plot(equilibrium_thin_surface); title('Equilibrium Thin Surface Input');
subplot(3, 2, 4); plot(equilibrium_output_thin_surface); title('Equilibrium Thin Surface Output');
subplot(3, 2, 5); plot(equilibrium_thick_surface); title('Equilibrium Thick Surface Input');
subplot(3, 2, 6); plot(equilibrium_output_thick_surface); title('Equilibrium Thick Surface Output');

% Function Definition (Moved to the end)
function output = spatio_temporal_filter(input_signal, spatial_kernel, temporal_kernel)
    output_spatial = zeros(size(input_signal));
    for t = 1:size(input_signal, 2)
        input_padded = [zeros((length(spatial_kernel)-1)/2, 1); input_signal(:, t); zeros((length(spatial_kernel)-1)/2, 1)];
        for x = 1:size(input_signal, 1)
            convolution_result = sum(input_padded(x:x+length(spatial_kernel)-1) .* spatial_kernel);
            output_spatial(x, t) = convolution_result(1); % Force scalar assignment
        end
    end
    output = zeros(size(input_signal));
    for x = 1:size(input_signal, 1)
        output(x, :) = conv(output_spatial(x, :), temporal_kernel, 'same');
    end
end



%%

% 2D Spatio-Temporal Edge/Line/Surface Detection

% Clear workspace and close figures.
clear;
close all;

% Parameters (Adjustable)
spatial_width = 21; % Spatial extent of kernel
spatial_height = 21;
temporal_length = 5; % Temporal extent
spatial_sigma_center = 2; % Center Gaussian sigma (spatial)
spatial_sigma_surround = 5; % Surround Gaussian sigma (spatial)
temporal_sigma = 1; % Temporal Gaussian sigma

% Create 2D Spatial Kernels (On-Center/Off-Surround)
[x, y] = meshgrid(-(spatial_width-1)/2:(spatial_width-1)/2, -(spatial_height-1)/2:(spatial_height-1)/2);
spatial_center = exp(-(x.^2 + y.^2) / (2 * spatial_sigma_center^2));
spatial_surround = exp(-(x.^2 + y.^2) / (2 * spatial_sigma_surround^2));
spatial_kernel = spatial_center - spatial_surround * (spatial_sigma_center / spatial_sigma_surround);
spatial_kernel = spatial_kernel / sum(abs(spatial_kernel(:))); % Normalize

% Create Temporal Kernel (Gaussian)
temporal_kernel = exp(-(0:temporal_length-1).^2 / (2 * temporal_sigma^2));
temporal_kernel = temporal_kernel / sum(temporal_kernel);

% Input Signals (2D Line, Thin Surface, Thick Surface)
input_line = zeros(100, 100, temporal_length);
input_line(50, :, :) = 1; % Line along y-axis

input_thin_surface = zeros(100, 100, temporal_length);
input_thin_surface(48:52, :, :) = 1; % Thin surface along y-axis

input_thick_surface = zeros(100, 100, temporal_length);
input_thick_surface(40:60, :, :) = 1; % Thick surface along y-axis

% Apply the Filter to Each Input
output_line = spatio_temporal_filter_2d(input_line, spatial_kernel, temporal_kernel);
output_thin_surface = spatio_temporal_filter_2d(input_thin_surface, spatial_kernel, temporal_kernel);
output_thick_surface = spatio_temporal_filter_2d(input_thick_surface, spatial_kernel, temporal_kernel);

% Plotting Spatio-Temporal Responses
figure;
subplot(3, 3, 1); imagesc(input_line(:, :, 1)); title('Input Line');
subplot(3, 3, 2); imagesc(output_line(:, :, 1)); title('Output Line');
subplot(3, 3, 4); imagesc(input_thin_surface(:, :, 1)); title('Input Thin Surface');
subplot(3, 3, 5); imagesc(output_thin_surface(:, :, 1)); title('Output Thin Surface');
subplot(3, 3, 7); imagesc(input_thick_surface(:, :, 1)); title('Input Thick Surface');
subplot(3, 3, 8); imagesc(output_thick_surface(:, :, 1)); title('Output Thick Surface');

% Equilibrium State (Static Input)
equilibrium_line = zeros(100, 100); equilibrium_line(50, :) = 1;
equilibrium_thin_surface = zeros(100, 100); equilibrium_thin_surface(48:52, :) = 1;
equilibrium_thick_surface = zeros(100, 100); equilibrium_thick_surface(40:60, :) = 1;

equilibrium_output_line = conv2(equilibrium_line, spatial_kernel, 'same');
equilibrium_output_thin_surface = conv2(equilibrium_thin_surface, spatial_kernel, 'same');
equilibrium_output_thick_surface = conv2(equilibrium_thick_surface, spatial_kernel, 'same');

% Plotting Equilibrium Responses
figure;
subplot(3, 2, 1); imagesc(equilibrium_line); title('Equilibrium Line Input');
subplot(3, 2, 2); imagesc(equilibrium_output_line); title('Equilibrium Line Output');
subplot(3, 2, 3); imagesc(equilibrium_thin_surface); title('Equilibrium Thin Surface Input');
subplot(3, 2, 4); imagesc(equilibrium_output_thin_surface); title('Equilibrium Thin Surface Output');
subplot(3, 2, 5); imagesc(equilibrium_thick_surface); title('Equilibrium Thick Surface Input');
subplot(3, 2, 6); imagesc(equilibrium_output_thick_surface); title('Equilibrium Thick Surface Output');

% Function Definition (Moved to the End)
function output = spatio_temporal_filter_2d(input_signal, spatial_kernel, temporal_kernel)
    output_spatial = zeros(size(input_signal));
    pad_x = (size(spatial_kernel, 1)-1)/2;
    pad_y = (size(spatial_kernel, 2)-1)/2;
    for t = 1:size(input_signal, 3)
        input_padded = zeros(size(input_signal, 1) + 2*pad_x, size(input_signal, 2) + 2*pad_y);
        input_padded(pad_x+1:end-pad_x, pad_y+1:end-pad_y) = input_signal(:, :, t); % Manual padding
        for x = 1:size(input_signal, 1)
            for y = 1:size(input_signal, 2)
                convolution_result = sum(sum(input_padded(x:x+size(spatial_kernel, 1)-1, y:y+size(spatial_kernel, 2)-1) .* spatial_kernel));
                output_spatial(x, y, t) = convolution_result(1); % Force scalar assignment
            end
        end
    end
    output = zeros(size(input_signal));
    for x = 1:size(input_signal, 1)
        for y = 1:size(input_signal, 2)
            output(x, y, :) = conv(reshape(output_spatial(x, y, :), 1, []), temporal_kernel, 'same'); % Reshape for conv
        end
    end
end

%%

% 2D Spatio-Temporal Edge/Line/Surface Detection with Temporal Blur

% Clear workspace and close figures.
clear;
close all;

% Parameters (Adjustable)
spatial_width = 21; % Spatial extent of kernel
spatial_height = 21;
temporal_length = 5; % Temporal extent
spatial_sigma_center = 2; % Center Gaussian sigma (spatial)
spatial_sigma_surround = 5; % Surround Gaussian sigma (spatial)
temporal_sigma = 1; % Temporal Gaussian sigma
bold_sigma = 2; % Sigma for BOLD-like temporal blurring

% Create 2D Spatial Kernels (On-Center/Off-Surround)
[x, y] = meshgrid(-(spatial_width-1)/2:(spatial_width-1)/2, -(spatial_height-1)/2:(spatial_height-1)/2);
spatial_center = exp(-(x.^2 + y.^2) / (2 * spatial_sigma_center^2));
spatial_surround = exp(-(x.^2 + y.^2) / (2 * spatial_sigma_surround^2));
spatial_kernel = spatial_center - spatial_surround * (spatial_sigma_center / spatial_sigma_surround);
spatial_kernel = spatial_kernel / sum(abs(spatial_kernel(:))); % Normalize

% Create Temporal Kernels
temporal_kernel = exp(-(0:temporal_length-1).^2 / (2 * temporal_sigma^2));
temporal_kernel = temporal_kernel / sum(temporal_kernel);
bold_kernel = exp(-(0:temporal_length-1).^2 / (2 * bold_sigma^2));
bold_kernel = bold_kernel / sum(bold_kernel);

% Input Signals (2D Line, Thin Surface, Thick Surface)
input_line = zeros(100, 100, temporal_length);
input_line(50, :, :) = 1; % Line along y-axis

input_thin_surface = zeros(100, 100, temporal_length);
input_thin_surface(48:52, :, :) = 1; % Thin surface along y-axis

input_thick_surface = zeros(100, 100, temporal_length);
input_thick_surface(40:60, :, :) = 1; % Thick surface along y-axis

% Apply the Filter to Each Input
output_line = spatio_temporal_filter_2d(input_line, spatial_kernel, temporal_kernel);
output_thin_surface = spatio_temporal_filter_2d(input_thin_surface, spatial_kernel, temporal_kernel);
output_thick_surface = spatio_temporal_filter_2d(input_thick_surface, spatial_kernel, temporal_kernel);

% Apply BOLD-like Temporal Blur
blurred_output_line = zeros(size(output_line));
blurred_output_thin_surface = zeros(size(output_thin_surface));
blurred_output_thick_surface = zeros(size(output_thick_surface));

for x = 1:size(output_line, 1)
    for y = 1:size(output_line, 2)
        blurred_output_line(x, y, :) = conv(reshape(output_line(x, y, :), 1, []), bold_kernel, 'same');
        blurred_output_thin_surface(x, y, :) = conv(reshape(output_thin_surface(x, y, :), 1, []), bold_kernel, 'same');
        blurred_output_thick_surface(x, y, :) = conv(reshape(output_thick_surface(x, y, :), 1, []), bold_kernel, 'same');
    end
end

% Equilibrium State (Static Input)
equilibrium_line = zeros(100, 100); equilibrium_line(50, :) = 1;
equilibrium_thin_surface = zeros(100, 100); equilibrium_thin_surface(48:52, :) = 1;
equilibrium_thick_surface = zeros(100, 100); equilibrium_thick_surface(40:60, :) = 1;

equilibrium_output_line = conv2(equilibrium_line, spatial_kernel, 'same');
equilibrium_output_thin_surface = conv2(equilibrium_thin_surface, spatial_kernel, 'same');
equilibrium_output_thick_surface = conv2(equilibrium_thick_surface, spatial_kernel, 'same');

% Plotting Comparison
figure;
subplot(3, 3, 1); imagesc(blurred_output_line(:, :, 1)); title('Blurred Line Output');
subplot(3, 3, 2); imagesc(equilibrium_output_line); title('Equilibrium Line Output');
subplot(3, 3, 4); imagesc(blurred_output_thin_surface(:, :, 1)); title('Blurred Thin Surface Output');
subplot(3, 3, 5); imagesc(equilibrium_output_thin_surface); title('Equilibrium Thin Surface Output');
subplot(3, 3, 7); imagesc(blurred_output_thick_surface(:, :, 1)); title('Blurred Thick Surface Output');
subplot(3, 3, 8); imagesc(equilibrium_output_thick_surface); title('Equilibrium Thick Surface Output');

% Function Definition (Moved to the End)
function output = spatio_temporal_filter_2d(input_signal, spatial_kernel, temporal_kernel)
    output_spatial = zeros(size(input_signal));
    pad_x = (size(spatial_kernel, 1)-1)/2;
    pad_y = (size(spatial_kernel, 2)-1)/2;
    for t = 1:size(input_signal, 3)
        input_padded = zeros(size(input_signal, 1) + 2*pad_x, size(input_signal, 2) + 2*pad_y);
        input_padded(pad_x+1:end-pad_x, pad_y+1:end-pad_y) = input_signal(:, :, t); % Manual padding
        for x = 1:size(input_signal, 1)
            for y = 1:size(input_signal, 2)
                convolution_result = sum(sum(input_padded(x:x+size(spatial_kernel, 1)-1, y:y+size(spatial_kernel, 2)-1) .* spatial_kernel));
                output_spatial(x, y, t) = convolution_result(1); % Force scalar assignment
            end
        end
    end
    output = zeros(size(input_signal));
    for x = 1:size(input_signal, 1)
        for y = 1:size(input_signal, 2)
            output(x, y, :) = conv(reshape(output_spatial(x, y, :), 1, []), temporal_kernel, 'same'); % Reshape for conv
        end
    end
end

%%


% 2D Spike-Triggered Cross-Correlation (Simplified)

% Clear workspace and close figures.
clear;
close all;

% Parameters (Adjustable)
spatial_width = 21;
spatial_height = 21;
spatial_sigma_center = 2;
spatial_sigma_surround = 5;
threshold = 0.1;

% Create 2D Spatial Kernel (On-Center/Off-Surround)
[x, y] = meshgrid(-(spatial_width - 1) / 2:(spatial_width - 1) / 2, -(spatial_height - 1) / 2:(spatial_height - 1) / 2);
spatial_center = exp(-(x.^2 + y.^2) / (2 * spatial_sigma_center^2));
spatial_surround = exp(-(x.^2 + y.^2) / (2 * spatial_sigma_surround^2));
spatial_kernel = spatial_center - spatial_surround * (spatial_sigma_center / spatial_sigma_surround);
spatial_kernel = spatial_center - spatial_surround .* (spatial_sigma_center / spatial_sigma_surround); % Corrected line
spatial_kernel = spatial_kernel / sum(abs(spatial_kernel(:)));

% Input Signals (Point Stimulus)
input_stimulus = zeros(100, 100);
stimulus_x = 50;
stimulus_y = 50;
input_stimulus(stimulus_x, stimulus_y) = 1;

% Apply the Spatial Filter
output_response = conv2(input_stimulus, spatial_kernel, 'same');

% "Spike" Detection (Thresholding)
spikes = output_response > threshold;

% Visualize Filtered Output and Spikes
figure;
subplot(1, 2, 1);
imagesc(output_response);
title('Filtered Output');
colorbar;

subplot(1, 2, 2);
imagesc(spikes);
title('Spikes');
colorbar;

% Receptive Field (Directly use the spatial kernel)
receptive_field = spatial_kernel;

% Plot Receptive Field
figure;
imagesc(receptive_field);
title('Receptive Field (Spatial Kernel)');
colorbar;


%%

 % 2D Second-Order Response (Two-Point Stimulus)

% Clear workspace and close figures.
clear;
close all;

% Parameters (Adjustable)
spatial_width = 21;
spatial_height = 21;
spatial_sigma_center = 2;
spatial_sigma_surround = 5;

% Create 2D Spatial Kernel (On-Center/Off-Surround)
[x, y] = meshgrid(-(spatial_width - 1) / 2:(spatial_width - 1) / 2, -(spatial_height - 1) / 2:(spatial_height - 1) / 2);
spatial_center = exp(-(x.^2 + y.^2) / (2 * spatial_sigma_center^2));
spatial_surround = exp(-(x.^2 + y.^2) / (2 * spatial_sigma_surround^2));
spatial_kernel = spatial_center - spatial_surround .* (spatial_sigma_center / spatial_sigma_surround);
spatial_kernel = spatial_kernel / sum(abs(spatial_kernel(:)));

% Stimulus Parameters
stimulus_center_x = 50; % Center stimulus x-coordinate
stimulus_center_y = 50; % Center stimulus y-coordinate
max_distance = 40; % Maximum distance for the second stimulus

% Distances to Test
distances = 0:max_distance;
response_values = zeros(size(distances));

% Loop Through Distances
for i = 1:length(distances)
    distance = distances(i);

    % Create Two-Point Stimulus
    input_stimulus = zeros(100, 100);
    input_stimulus(stimulus_center_x, stimulus_center_y) = 1;

    % Place the second point at the calculated distance
    stimulus_second_x = stimulus_center_x + distance;

    % Make sure second stimulus is within bounds
    if stimulus_second_x > 100
        stimulus_second_x = 100;
    end

    input_stimulus(stimulus_second_x, stimulus_center_y) = 1;

    % Apply Spatial Filter
    output_response = conv2(input_stimulus, spatial_kernel, 'same');

    % Get the response at the center stimulus location
    response_values(i) = output_response(stimulus_center_x, stimulus_center_y);
end

% Plot Second-Order Response
figure;
plot(distances, response_values);
title('Second-Order Response (Two-Point Stimulus)');
xlabel('Distance from Center Stimulus');
ylabel('Response at Center');
grid on;


%%
