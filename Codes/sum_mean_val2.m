function   vec = sum_mean_val2(input, K)
% From Bouboulis et al. code

N=length(input);
for i=1:N
    vec(i)=mean(input(max(1,i-K+1):i));
end;