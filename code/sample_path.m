function code = sample_path(ini_state,transition,duration,T,scheme)
% assume Poisson duration distribution with duration rounded to
if nargin < 5
    scheme = 1; % most possible transition
end
count = 0;
code = zeros(1,2*T);
% duration = round(duration);  %%%%%%%%%% why need this?
while count < T
    t = poissrnd(duration(ini_state));
    code(count+1:count+t) = ini_state;
    if scheme == 1
        [~,ini_state] = max(transition(ini_state,:)); % next most possible state
    else
        ini_state = sample_discrete(transition(ini_state,:)); % random selection
    end
    count = count + t;
end
code = code(1:T);