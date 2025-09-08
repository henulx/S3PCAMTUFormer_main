clc
clear
close all

database = 'Houston';

load('.\Houstonlabel.mat');

gth = Houstonlabel;

IterNum = 10;
randp = randpGen(gth,IterNum);
save([database '_gt_randp.mat']);




