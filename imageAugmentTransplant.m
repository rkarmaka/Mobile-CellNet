function [T] = imageAugmentTransplant()
    % this function takes the path of a folder as an input and takes all
    % the images on the folder.
    num = 5;
    outDim = 512;
    windSize = 512;
    D = uigetdir;
    S1 = dir(fullfile(D,'*.jpg'));
    fnum = 1;
    T = table(rand(1),rand(1), 'VariableNames',{'OldName','NewName'});
       
    for k = 1:numel(S1)
        F1 = fullfile(D,S1(k).name);
        I = imread(F1);
        I = convert_gray(I);
        for i = 1:num
            [imAug] = hexa(outDim,windSize,I,1);
            fname = string(randi([11, 999999999]))+'.tiff';
            s1=split(S1(k).name,'.');
            Tnew = table(s1(1), fname, 'VariableNames',{'OldName','NewName'});
            T = [T;Tnew];
            folder = 'D:\RMLEB - Konan\Images\Dataset\Train\Transplant';
            fullFileName = fullfile(folder, fname);
            imwrite(imAug, fullFileName);
            
        end

     fnum = fnum+1;
    end
    
end