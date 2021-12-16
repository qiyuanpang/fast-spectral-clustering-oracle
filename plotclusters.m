close all; clear all; clc;

n_samples = 10000
n = 100
m = 100
fname = "labels_" + num2str(n_samples) + "_2_2_";

fid = fopen(fname+"pos.json", 'r');
raw = fread(fid, inf);
str = char(raw');
labels1 = jsondecode(str);
labels1 = labels1.labels;

fid = fopen(fname+"bin.json", 'r');
raw = fread(fid, inf);
str = char(raw');
labels2 = jsondecode(str);
labels2 = labels2.labels;

fid = fopen(fname+"abs.json", 'r');
raw = fread(fid, inf);
str = char(raw');
labels3 = jsondecode(str);
labels3 = labels3.labels;

l1 = labels1(1);
if l1
    idx = find(labels1==0);
    labels1(labels1==1) = 0;
    labels1(idx) = 1;
end

l2 = labels2(1);
if l2
    idx = find(labels2==0);
    labels2(labels2==1) = 0;
    labels2(idx) = 1;
end

l3 = labels3(1);
if l3
    idx = find(labels3==0);
    labels3(labels3==1) = 0;
    labels3(idx) = 1;
end

A1 = reshape(labels1, n, m);
A2 = reshape(labels2, n, m);
A3 = reshape(labels3, n, m);

ratio1 = round(length(find(labels1==0))/n_samples,2);
ratio2 = round(length(find(labels2==0))/n_samples,2);
ratio3 = round(length(find(labels3==0))/n_samples,2);

fig = figure()
subplot(1,3,1);
image(A1*255);hold on;
axis square;
title("posSim("+num2str(ratio1)+","+num2str(1-ratio1)+")")

subplot(1,3,2);
image(A2*255);hold on;
axis square;
title("binSim("+num2str(ratio2)+","+num2str(1-ratio2)+")")

subplot(1,3,3);
image(A3*255);hold on;
axis square;
title("absDiagonal("+num2str(ratio3)+","+num2str(1-ratio3)+")")

savefile = "cluster_2_" + num2str(n_samples) + ".png";
saveas(fig, savefile);
exit
