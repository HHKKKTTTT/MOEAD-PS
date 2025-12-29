classdef MOEAD_PS < ALGORITHM
    methods
        function main(Algorithm,Problem)
            %% 参数设置
            % 生成正态分布方向向量
            Problem.N = 100;           % 种群大小
            M = Problem.M;             % 目标维度
            D = Problem.D;
            r1 = normrnd(0.5,0.07,[Problem.N,1]);
            r2 = 1 - r1;
            r = zeros(Problem.N, M+1);
            r(:,1) = r1;
            r(:,2) = r2;
            r(1, 1:M+1) = (r(1,1)+r(1,2))/2;
            rs1 = r(:,1);
            [~,indexRs1]=sort(rs1);
            pomin=indexRs1(1);
            pomax=indexRs1(100);
            tanmax=r(pomin,2)/r(pomin,1);
            arctmin=atan(r(pomax,2)/r(pomax,1));
            k = 0.01;


            %% 邻居参数设置
            T = ceil(Problem.N/10);     % 邻居大小
            [~,B] = sort(pdist2(r,r),2); % 计算方向向量相似度
            B = B(:,1:T);              % 获取邻居索引
            
            %% 初始化种群
            Population = Problem.Initialization();
            
            it = 1;
            %% 优化主循环
             while Algorithm.NotTerminated(Population)

                   % 获取种群约束和目标信息
                    Cons = sum(max(0,Population.cons),2); % 约束违反值      
                    Objs = [Population.objs, Cons];  % 初始种群目标信息
                    feasible = Cons <= 0;
                    % 约束排序
                    [~,sortIdx] = sort(Cons,'descend');
                    remainIdx = sortIdx(1:end);
                    % 获取所有解的约束值
                    consValues = Cons(remainIdx);
                    % 找到最小约束值
                    [minCons, ~] = min(consValues);
                    % 找出所有等于最小约束值的索引（在remainIdx中的位置）
                    candidates = find(consValues == minCons);
                     % 在这些候选中找到目标函数最小的索引
                    [~, minObjIdx] = min(Objs(remainIdx(candidates), 1));
                    gMinIdx = candidates(minObjIdx);
                    f2 = Objs(remainIdx(gMinIdx), 1);
                    g2 = Cons(remainIdx(gMinIdx));

                   
                    % 找出所有约束值在 [minCons, minCons + k] 范围内的个体
                    candidateIndices = find(consValues >= minCons & consValues <= minCons + k);
                    if ~isempty(candidateIndices)
                        % 在候选个体中找到目标函数最小的个体
                        candidateObjs = Objs(remainIdx(candidateIndices), 1); % 提取目标函数值
                        [minObj, minObjIdx] = min(candidateObjs);
                        if minObj <= f2
                        GMinIdx = candidateIndices(minObjIdx);
                        f1 = Objs(remainIdx(GMinIdx), 1);
                        g1 = Cons(remainIdx(GMinIdx));
                        x0 = (tanmax*f1 - f2 + g2 - g1)/(tanmax-1);
                        y = f2-x0;
                        end
                    end
                    
                   
                    objs_P = [[Population.obj]',Cons];
                    temp = objs_P-[f2,g2];
                    delta = max(temp(:,1))-min(temp(:,1));
                    temprel = temp(:,2)./temp(:,1);
                    mintemprel = min(temprel);
                    if mintemprel < 0
                        minusidx = temprel<0;
                        mi = minusidx==1;
                        minus = temp(mi,:);
                        normMinus = arrayfun(@(x)norm(minus(x,:)),1:size(minus,1));
                        [~,normMinusidx] = max(normMinus);
                        tempindex = temp==minus(normMinusidx);
                        fg1 = temp(tempindex(:,1),:);
                        x = abs(fg1(1));
                    else
                        x= (sqrt(2)*sin(arctmin)*delta)/(2*sin((pi/4)-arctmin));
                    end
                    
                    if abs(y) < x
                        a = [f2,g2]-x;
                    else
                        a = [f2,g2]-abs(y);
                    end                  
                
                    if it < 0.5 * Problem.maxFE/100
                        Parameter={1,0.1,1,0.1};
                        k=0.05;
                    elseif it < 0.7 * Problem.maxFE/100
                        Parameter={1,10,max(1,round(D*0.4)),10};
                        k=0.001;
                    elseif it < 0.8 * Problem.maxFE/100
                        Parameter={1,100,max(1,round(D*0.5)),100};
                        k=0;
                    else
                        Parameter={1,10000,max(1,round(D*0.5)),10000}; 
                    end
               

               %% 个体更新
                for i = 1:Problem.N
                    % 选择邻居
                    P = B(i,randperm(size(B,2)));
                    % 遗传操作生成子代
                    Offspring = OperatorGAhalf(Problem,Population(P(1:2)),Parameter);
                    offspringCons = sum(max(0,Offspring.con));
                    offspringObjsCons = [Offspring.obj, offspringCons];
                    if (Offspring.obj < f2 && offspringCons == g2)||offspringCons < g2
                        f2 = Offspring.obj;
                        g2 = offspringCons;
                        pCons  = sum(max(0,Population.cons),2); 
                        objs_P = [[Population.obj]',pCons];
                     
                        ifcandidateIndices = find(pCons >= g2 & pCons <= g2 + k);
                        candidateObjs = objs_P(ifcandidateIndices, 1); % 提取目标函数值
                        [ifminObj, ifminObjIdx] = min(candidateObjs);
                        if ifminObj < f2
                            ifGMinIdx = ifcandidateIndices(ifminObjIdx);
                            iff1 = objs_P(ifGMinIdx, 1);
                            ifg1 = objs_P(ifGMinIdx);
                            ifx0 = (tanmax*iff1 - f2 + g2 - ifg1)/(tanmax-1);
                            ify = f2 - ifx0;
                        else
                            ify=0;
                        end
                        
                        tempif   = objs_P-repmat(offspringObjsCons,Problem.N,1);
                        deltaif = max(tempif(:,1))-min(tempif(:,1));
                        tempifrel = tempif(:,2)./tempif(:,1);
                        mintempifrel = min(tempifrel);
                        if mintempifrel < 0
                            minusifidx = tempifrel<0;
                            mif = minusifidx==1;
                            minusif = tempif(mif,:);
                            normMinus = arrayfun(@(mif)norm(minusif(mif,:)),1:size(minusif,1));
                            [~,normMinusifidx] = max(normMinus);
                            tempifindex = tempif==minusif(normMinusifidx);
                            fg1 = tempif(tempifindex(:,1),:);
                            xif = abs(fg1(1));
                        else
                            xif=(sqrt(2)*sin(arctmin)*deltaif)/(2*sin((pi/4)-arctmin));
                        end
                        
                        if abs(ify)<xif
                            a = offspringObjsCons-xif;
                        else
                            a = offspringObjsCons-abs(ify);
                        end
                    
                        tempif   = abs(objs_P-repmat(a,Problem.N,1));
                        temp_population   = Population;
                        temp_population(1)  = Offspring;
                        for jj=2:Problem.N
                             [~,bestindex] = min(max(tempif./r(jj,:),[],2));
                             temp_population(jj) = Population(bestindex);
                             tempif(bestindex,:)=1e+9;   
                        end
                        Population = temp_population;
                    else
                    % 当前个体的目标约束组合
                    pCons = sum(max(0,Population(P).cons),2); 
                    objs_P = [[Population(P).obj]',pCons]; 
                    
                    %% 切比雪夫计算方法
                    g_old = max(abs(objs_P-repmat(a,T,1))./r(P,:),[],2);  
                    g_new = max(repmat(abs(offspringObjsCons-a),T,1)./r(P,:),[],2);
                    
                    %% 更新条件判断（增加约束违反度判断）
                    Population(P(g_old>=g_new)) = Offspring;
                    end
                end
                it = it+1;
            end
        end
    end
end
