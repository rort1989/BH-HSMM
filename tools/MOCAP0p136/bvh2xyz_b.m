function xyz = bvh2xyz_b(skel, channels)

% WARNING:
% This is a surrogate for the bvh2xyz.m file that comes with the MATLAB
% Motion Capture Toolbox of Prof. Neil Lawrence. This surrogate fixes a bug
% in computation of XYZ coordinates from the BVH files available in the
% Berkeley MHAD. 2013-05-29
%
% BVH2XYZ Compute XYZ values given structure and channels.
%
%	Description:
%
%	XYZ = BVH2XYZ(SKEL, CHANNELS) Computes X, Y, Z coordinates given a
%	BVH skeleton structure and an associated set of channels.
%	 Returns:
%	  XYZ - the point cloud positions for the skeleton.
%	 Arguments:
%	  SKEL - a skeleton for the bvh file.
%	  CHANNELS - the channels for the bvh file.
%	
%
%	See also
%	ACCLAIM2XYZ, SKEL2XYZ


%	Copyright (c) 2005, 2008 Neil D. Lawrence
% 	bvh2xyz.m CVS version 1.3
% 	bvh2xyz.m SVN version 42
% 	last update 2008-08-12T20:23:47.000000Z

for i = 1:length(skel.tree)  
  if ~isempty(skel.tree(i).posInd)
    xpos = channels(skel.tree(i).posInd(1));
    ypos = channels(skel.tree(i).posInd(2));
    zpos = channels(skel.tree(i).posInd(3));
    thisPosition = [xpos ypos zpos];
  else
    thisPosition = skel.tree(i).offset;
  end
  xyzStruct(i) = struct('rotation', [], 'xyz', []); 
  if nargin < 2 || isempty(skel.tree(i).rotInd)
    xangle = 0;
    yangle = 0;
    zangle = 0;
  else
    xangle = deg2rad(channels(skel.tree(i).rotInd(1)));
    yangle = deg2rad(channels(skel.tree(i).rotInd(2)));
    zangle = deg2rad(channels(skel.tree(i).rotInd(3)));
  end
  thisRotation = rotationMatrix(xangle, yangle, zangle, skel.tree(i).order);
  if ~skel.tree(i).parent
    xyzStruct(i).rotation = thisRotation;
    xyzStruct(i).xyz = thisPosition;
  else
    xyzStruct(i).xyz = ...
    thisPosition * xyzStruct(skel.tree(i).parent).rotation ...
        + xyzStruct(skel.tree(i).parent).xyz;
    xyzStruct(i).rotation = thisRotation*xyzStruct(skel.tree(i).parent).rotation;    
  end
end
xyz = reshape([xyzStruct(:).xyz], 3, length(skel.tree))';



