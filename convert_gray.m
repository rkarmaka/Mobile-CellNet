function im_gray = convert_gray(image)
    [x, y, dim] = size(image);
    if dim ==3
        im_gray = rgb2gray(image);
    else
        im_gray = image;
    end
    %figure;
    if dim == 3
        %subplot(121);imshow(image,[]);title('Input Image --> RGB');
        %subplot(122);imshow(im_gray,[]);title('Input Image --> Gray Scale');
    else
        %title('Input Image --> Gray Scale');imshow(image,[]);
    end
end