

% Parameters (Adjusted for Rat Vision - Based on published research)
spatial_width = 15;
spatial_height = 15;
temporal_length = 7;
spatial_sigma_center = 1;
spatial_sigma_surround = 3;
temporal_sigma = 1.5;

% Create 2D Spatial Kernels
[x, y] = meshgrid(-(spatial_width-1)/2:(spatial_width-1)/2, -(spatial_height-1)/2:(spatial_height-1)/2);
spatial_center = exp(-(x.^2 + y.^2) / (2 * spatial_sigma_center^2));
spatial_surround = exp(-(x.^2 + y.^2) / (2 * spatial_sigma_surround^2));
spatial_kernel = spatial_center - spatial_surround * (spatial_sigma_center / spatial_sigma_surround);
spatial_kernel = spatial_kernel / sum(abs(spatial_kernel(:)));

% Create Temporal Kernel
temporal_kernel = exp(-(0:temporal_length-1).^2 / (2 * temporal_sigma^2));
temporal_kernel = temporal_kernel / sum(temporal_kernel);

% Input Signals
input_line = zeros(100, 100, temporal_length);
input_line(50, :, :) = 1;
input_thin_surface = zeros(100, 100, temporal_length);
input_thin_surface(48:52, :, :) = 1;
input_thick_surface = zeros(100, 100, temporal_length);
input_thick_surface(40:60, :, :) = 1;

% Apply the temporal Filter
output_line = spatio_temporal_filter_2d(input_line, spatial_kernel, temporal_kernel);
output_thin_surface = spatio_temporal_filter_2d(input_thin_surface, spatial_kernel, temporal_kernel);
output_thick_surface = spatio_temporal_filter_2d(input_thick_surface, spatial_kernel, temporal_kernel);

% Plotting Spatio-Temporal Responses
figure;
subplot(3, 3, 1); imagesc(input_line(:, :, 1)); title('Input Line');
subplot(3, 3, 2); imagesc(output_line(:, :, 1)); title('Output Line (Rat)');
subplot(3, 3, 4); imagesc(input_thin_surface(:, :, 1)); title('Input Thin Surface');
subplot(3, 3, 5); imagesc(output_thin_surface(:, :, 1)); title('Output Thin Surface (Rat)');
subplot(3, 3, 7); imagesc(input_thick_surface(:, :, 1)); title('Input Thick Surface');
subplot(3, 3, 8); imagesc(output_thick_surface(:, :, 1)); title('Output Thick Surface (Rat)');

% Equilibrium State
equilibrium_line = zeros(100, 100); equilibrium_line(50, :) = 1;
equilibrium_thin_surface = zeros(100, 100); equilibrium_thin_surface(48:52, :) = 1;
equilibrium_thick_surface = zeros(100, 100); equilibrium_thick_surface(40:60, :) = 1;

equilibrium_output_line = conv2(equilibrium_line, spatial_kernel, 'same');
equilibrium_output_thin_surface = conv2(equilibrium_thin_surface, spatial_kernel, 'same');
equilibrium_output_thick_surface = conv2(equilibrium_thick_surface, spatial_kernel, 'same');

% Plotting Equilibrium Responses
figure;
subplot(3, 2, 1); imagesc(equilibrium_line); title('Equilibrium Line Input');
subplot(3, 2, 2); imagesc(equilibrium_output_line); title('Equilibrium Line Output (Rat)');
subplot(3, 2, 3); imagesc(equilibrium_thin_surface); title('Equilibrium Thin Surface Input');
subplot(3, 2, 4); imagesc(equilibrium_output_thin_surface); title('Equilibrium Thin Surface Output (Rat)');
subplot(3, 2, 5); imagesc(equilibrium_thick_surface); title('Equilibrium Thick Surface Input');
subplot(3, 2, 6); imagesc(equilibrium_output_thick_surface); title('Equilibrium Thick Surface Output (Rat)');

% Function definition
function output = spatio_temporal_filter_2d(input_signal, spatial_kernel, temporal_kernel)
    output_spatial = zeros(size(input_signal));
    pad_x = floor((size(spatial_kernel, 1)-1)/2);
    pad_y = floor((size(spatial_kernel, 2)-1)/2);
    for t = 1:size(input_signal, 3)
        input_padded = zeros(size(input_signal, 1) + 2*pad_x, size(input_signal, 2) + 2*pad_y);
        input_padded(pad_x+1:end-pad_x, pad_y+1:end-pad_y) = input_signal(:, :, t);
        for x = 1:size(input_signal, 1)
            for y = 1:size(input_signal, 2)
                row_start = x;
                row_end = x + size(spatial_kernel, 1) - 1;
                col_start = y;
                col_end = y + size(spatial_kernel, 2) - 1;
                convolution_region = input_padded(row_start:row_end, col_start:col_end);
                convolution_result = sum(sum(convolution_region .* spatial_kernel));
                output_spatial(x, y, t) = convolution_result(1);
            end
        end
    end
    output = zeros(size(input_signal));
    for x = 1:size(input_signal, 1)
        for y = 1:size(input_signal, 2)
            output(x, y, :) = conv(reshape(output_spatial(x, y, :), 1, []), temporal_kernel, 'same');
        end
    end
end