% https://in.mathworks.com/help/deeplearning/ref/importnetworkfrompytorch.html
% ^ Official MATLAB documentation for importing models from pyTorch
modelfile = './traced_   .pt';

net = importNetworkFromPyTorch(modelfile);

InputSize = [224 224 3];
inputLayer = imageInputLayer(InputSize,Normalization="none");
net = addInputLayer(net,inputLayer,Initialize=true); %pytorch model did not need an input layer

%%
test_img = imread("./data/imagenette2-160/val/n02102040/ILSVRC2012_val_00025442.JPEG"); % english springer
test_img = imresize(test_img, InputSize(1:2));
imshow(test_img);

%%
test_img = rescale(test_img,0,1);

meanIm = [0.485 0.456 0.406];
stdIm = [0.229 0.224 0.225];
test_img = (test_img - reshape(meanIm,[1 1 3]))./reshape(stdIm,[1 1 3]); % preprocessing an ImageNet-pretrained model needs

Im_dlarray = dlarray(single(test_img),"SSCB");

%%
[~,ClassNames] = imagePretrainedNetwork("resnet18");
prob = predict(net,Im_dlarray);
[sorted_probs, sorted_inds] = sort(prob, 'descend');

for i=1:5
    fprintf('Pred %d: %s (%f) \n', i, ClassNames(sorted_inds(i)),sorted_probs(i) );
end


