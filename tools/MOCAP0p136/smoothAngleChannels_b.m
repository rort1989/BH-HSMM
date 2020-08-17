function channels = smoothAngleChannels_b(channels, skel)

% SMOOTHANGLECHANNELS Try and remove artificial discontinuities associated with angles.
%
%	Description:
%	channels = smoothAngleChannels(channels, skel);
%% 	smoothAngleChannels.m CVS version 1.1
% 	smoothAngleChannels.m SVN version 30
% 	last update 2008-01-12T11:32:50.000000Z
for col = 1:size(channels,2)
% % for i=1:length(skel.tree)
% %   for j=1:length(skel.tree(i).rotInd)    
% %     col = skel.tree(i).rotInd(j);
% %     if col
%       for k=2:size(channels, 1)
%         diff=channels(k, col)-channels(k-1, col);
%         if diff > 270          
%             channels(k, col) = channels(k, col) - 360;
%         elseif diff < -270
%             channels(k, col) = channels(k, col) + 360;
%         end                
%       end
% %     end
% %   end
% % end
    for k=1:size(channels, 1)
        if channels(k, col) < -173
            channels(k, col) = channels(k, col) + 360;
        end
    end
end