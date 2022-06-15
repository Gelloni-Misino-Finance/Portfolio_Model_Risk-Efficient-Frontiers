function [data1]=DataCleaning(data1,eta,monotonicity)%,numbofclean,numbofclean
%Analysis of the step related to the single iteration

% Function used to reduce the noise, associated to the calibrated parameters, related to some optimizations' problems
%
% INPUT:
%
% theta:        vector of theta obtained by the calibration
% S:            vector of volatilities obtained by the calibration
% eta:          vector of eta (different values of KL ball's ray)
%
%
% OUTPUT:
%
% theta:        vector of cleaned theta (for the plot)
% S:            vector of cleaned volatilities 
%

%PENSARE IMPLEMENTAZIONE MONOTONIA.

%||(numbofclean>0)
  MT=mean(data1);
  idx=find((abs(data1)<(abs(MT)+std(data1))& ((abs(data1))>(abs(MT)-2*std(data1))))); 
  if monotonicity==1%Increasing
      idx2=find((data1(1:end-1)<data1(2:end))); 
      idx=intersect(idx,idx2);
  elseif monotonicity==-1%Decreasing
      idx2=find((data1(1:end-1)>data1(2:end))); 
      idx=intersect(idx,idx2);
  end
  curve=fit(eta(idx),data1(idx),'smoothingspline');
  notidx=ones(length(eta),1);
  notidx(idx)=0;
  notidx=find(notidx==1);
  %notidx=find ( ( abs(data1)>(abs(MT)+std(data1)) ) | ( abs(data1)<(abs(MT)-std(data1))) ); 
  data1(notidx)=feval(curve,eta(notidx));
  
  
%   numbofclean=numbofclean+1;
% else
% MT=mean(data1.^2);
% idx=find((abs(data1)<(abs(MT)+std(data1.^2))*((abs(data1))>(abs(MT)-2*std(data1.^2))))); 
% notidx=find ( ( abs(data1)>(abs(MT)+std(data1.^2)) ) | ( abs(data1)<(abs(MT)-std(data1.^2))) ); 
% if(isempty(idx))
%         MT=mean(data1);
%         idx=find((abs(data1)<(abs(MT)+std(data1))*((abs(data1))>(abs(MT)-2*std(data1))))); 
%         notidx=find ( ( abs(data1)>(abs(MT)+std(data1)) ) | ( abs(data1)<(abs(MT)-std(data1))) ); 
%         idx2=find((data1(1:end-1)>data1(2:end))); 
%         idx=intersect(idx,idx2);
% end
%   curve=fit(eta(idx),data1(idx),'smoothingspline');
%   data1(notidx)=feval(curve,eta(notidx));
%   numbofclean=numbofclean+1;
end

% for i=notidx
%         data1(i)=interp1(eta(idx),data1(idx),eta(i),'spline','extrap');
% end       
    
%     for i=1:length(eta)-1
%         if (abs(data1(i))>abs(MT))
%         data1(i)=interp1(eta(idx),data1(idx),eta(i),'spline','extrap');
%        
%         end



