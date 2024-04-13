function [out1, out2] = hexa(dim, side, img, mask)
    [a, b, d] = size(img);
    if d == 3
        img = rgb2gray(img);
    end
    tempX = a - ceil(side/2);
    tempY = b - ceil(side/2);
    x = randi([ceil(side/2),tempX]);
    y = randi([ceil(side/2),tempY]);
    out1 = img((x-(side/2)+1):(x-(side/2))+side, (y-(side/2)+1):(y-(side/2))+side);
    out1 = imresize(out1,[dim dim]);
%    out2 = mask((x-(side/2)+1):(x-(side/2))+side, (y-(side/2)+1):(y-(side/2))+side);
%    out2 = imresize(out2,[dim dim]);
end




