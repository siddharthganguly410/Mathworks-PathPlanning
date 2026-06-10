classdef RoboR2 < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure        matlab.ui.Figure
        % Left panel controls
        LeftPanel       matlab.ui.container.Panel
        TitleLabel      matlab.ui.control.Label
        EpLabel         matlab.ui.control.Label
        EpSpinner       matlab.ui.control.Spinner
        StepsLabel      matlab.ui.control.Label
        StepsSpinner    matlab.ui.control.Spinner
        AlphaLabel      matlab.ui.control.Label
        AlphaSpinner    matlab.ui.control.Spinner
        GammaLabel      matlab.ui.control.Label
        GammaSpinner    matlab.ui.control.Spinner
        R2Label         matlab.ui.control.Label
        R2Field         matlab.ui.control.EditField
        FakeLabel       matlab.ui.control.Label
        FakeSpinner     matlab.ui.control.Spinner
        RunBtn          matlab.ui.control.Button
        StatusLabel     matlab.ui.control.Label
        % Tab group
        TabGroup        matlab.ui.container.TabGroup
        Tab1            matlab.ui.container.Tab
        Tab2            matlab.ui.container.Tab
        Tab3            matlab.ui.container.Tab
        Tab4            matlab.ui.container.Tab
        Tab5            matlab.ui.container.Tab
        Tab6            matlab.ui.container.Tab
        % Axes
        AxReward        matlab.ui.control.UIAxes
        AxSteps         matlab.ui.control.UIAxes
        AxSuccess       matlab.ui.control.UIAxes
        AxForest        matlab.ui.control.UIAxes
        AxPath          matlab.ui.control.UIAxes
        % Summary labels
        SumPanel        matlab.ui.container.Panel
        SRTitle         matlab.ui.control.Label
        SRVal           matlab.ui.control.Label
        ASTitle         matlab.ui.control.Label
        ASVal           matlab.ui.control.Label
        MSTitle         matlab.ui.control.Label
        MSVal           matlab.ui.control.Label
        ARTitle         matlab.ui.control.Label
        ARVal           matlab.ui.control.Label
        EPTitle         matlab.ui.control.Label
        EPVal           matlab.ui.control.Label
        QSTitle         matlab.ui.control.Label
        QSVal           matlab.ui.control.Label
        BTTitle         matlab.ui.control.Label
        BTVal           matlab.ui.control.Label
    end

    % Colour palette
    properties (Access = private)
        C_ACCENT   = [0.000, 0.898, 1.000]
        C_SUCCESS  = [0.412, 1.000, 0.278]
        C_WARN     = [1.000, 0.667, 0.000]
        C_DANGER   = [1.000, 0.302, 0.427]
        C_BG       = [0.102, 0.114, 0.180]
        C_FIGBG    = [0.059, 0.067, 0.090]
        C_TEXT     = [0.878, 0.878, 0.878]
        C_GRID     = [0.165, 0.176, 0.243]
        C_SCROLL   = [1.000, 0.843, 0.000]
        C_FAKE     = [1.000, 0.302, 0.427]
        C_R1       = [0.706, 0.557, 0.678]
        C_EMPTY    = [0.165, 0.176, 0.243]
    end

    % Callbacks
    methods (Access = private)

        function RunButtonPushed(app, ~)
            app.RunBtn.Enable = 'off';
            app.StatusLabel.Text = 'Initialising...';
            drawnow;

            %% --- Constants ---
            ROWS = 4; COLS = 3; NB = 12;
            UP=1; DOWN=2; LEFT=3; RIGHT=4; COLLECT=5; EXIT_A=6;

            %% --- Config from UI ---
            EPISODES  = app.EpSpinner.Value;
            MAX_STEPS = app.StepsSpinner.Value;
            try
                r2_real = str2num(app.R2Field.Value); %#ok<ST2NM>
                if isempty(r2_real), r2_real = [1 7 5 11]; end
            catch
                r2_real = [1 7 5 11];
            end
            fake_scroll = app.FakeSpinner.Value;
            r1_scrolls  = [12 6 10];

            env_cfg.r2_real = r2_real;
            env_cfg.fake    = fake_scroll;
            env_cfg.r1      = r1_scrolls;

            %% --- Build adjacency ---
            ADJ = cell(NB,1);
            for b = 1:NB
                [r,c] = app.b2rc(b,COLS);
                nb = [];
                if r>0,      nb(end+1)=app.rc2b(r-1,c,COLS); end
                if r<ROWS-1, nb(end+1)=app.rc2b(r+1,c,COLS); end
                if c>0,      nb(end+1)=app.rc2b(r,c-1,COLS); end
                if c<COLS-1, nb(end+1)=app.rc2b(r,c+1,COLS); end
                ADJ{b} = nb;
            end

            %% --- Environment struct ---
            env.r2_real      = r2_real;
            env.fake         = fake_scroll;
            env.r1           = r1_scrolls;
            env.ROWS=ROWS; env.COLS=COLS; env.NB=NB;
            env.ADJ=ADJ;
            env.ENTRY=[1 2 3]; env.EXIT=[10 11 12];
            env.UP=UP; env.DOWN=DOWN; env.LEFT=LEFT;
            env.RIGHT=RIGHT; env.COLLECT=COLLECT; env.EXIT_A=EXIT_A;
            % Rewards
            env.R_REAL=100; env.R_ALL=500; env.R_EXIT=300;
            env.R_MOVE=-1;  env.R_EXTRA=-2; env.R_FAKEPROX=-500;
            env.R_FAKE=-1000; env.R_R1=-1000;
            env.R_INV=-500; env.R_WANDER=-5;
            env = app.envReset(env);

            %% --- Agent ---
            ag.alpha   = app.AlphaSpinner.Value;
            ag.gamma   = app.GammaSpinner.Value;
            ag.eps     = 1.0;
            ag.eps_min = 0.01;
            ag.eps_dec = 0.9997;
            ag.Q_opt   = 10.0;
            ag.Q       = containers.Map('KeyType','char','ValueType','any');
            ag.visits  = containers.Map('KeyType','char','ValueType','any');
            ag.rew     = [];
            ag.stp     = [];
            ag.succ    = [];

            %% --- Draw forest layout immediately ---
            app.drawForest(env_cfg, ROWS, COLS, NB);
            drawnow;

            %% --- Training ---
            VERB    = max(1, floor(EPISODES/10));
            PAT     = min(3000, EPISODES);
            sw      = zeros(1,PAT);
            sw_ptr  = 0; sw_n = 0;
            best_rew  = -inf;
            best_path = {};
            t0 = tic;

            for ep = 1:EPISODES
                env   = app.envReset(env);
                st    = app.getState(env);
                ep_r  = 0;
                succ  = false;

                for s = 1:MAX_STEPS
                    legal = app.legalActs(env);
                    if isempty(legal), break; end
                    act = app.selectAct(ag, st, legal, false);
                    [env, r, done, info] = app.step(env, act);
                    nst    = app.getState(env);
                    nlegal = app.legalActs(env);
                    ag     = app.updateQ(ag, st, act, r, nst, nlegal, done);
                    st     = nst;
                    ep_r   = ep_r + r;
                    if done
                        if isfield(info,'success') && info.success, succ=true; end
                        break;
                    end
                end

                ag.eps = max(ag.eps_min, ag.eps * ag.eps_dec);
                ag.rew(end+1)  = ep_r;
                ag.stp(end+1)  = env.steps;
                ag.succ(end+1) = double(succ);

                sw_ptr = mod(sw_ptr, PAT)+1;
                sw(sw_ptr) = double(succ);
                if sw_n < PAT, sw_n = sw_n+1; end

                if succ && ep_r > best_rew
                    best_rew  = ep_r;
                    best_path = env.path;
                end

                if mod(ep, VERB)==0
                    pct = round(ep/EPISODES*100);
                    app.StatusLabel.Text = sprintf('Training... %d%% (ep %d/%d)', pct, ep, EPISODES);
                    app.plotReward(ag);
                    app.plotSteps(ag);
                    app.plotSuccess(ag);
                    drawnow limitrate;
                end

                if sw_n==PAT && sum(sw)/PAT >= 0.98
                    break;
                end
            end
            elapsed = toc(t0);

            app.plotReward(ag);
            app.plotSteps(ag);
            app.plotSuccess(ag);

            %% --- Greedy evaluation ---
            app.StatusLabel.Text = 'Evaluating greedy policy...';
            drawnow;

            EVAL=200; EMAXS=100;
            res = struct('success',{},'steps',{},'reward',{},'path',{});
            for e = 1:EVAL
                env  = app.envReset(env);
                st   = app.getState(env);
                suc  = false;
                for s = 1:EMAXS
                    legal = app.legalActs(env);
                    if isempty(legal), break; end
                    act = app.selectAct(ag, st, legal, true);
                    [env,~,done,info] = app.step(env, act);
                    st = app.getState(env);
                    if done
                        if isfield(info,'success') && info.success, suc=true; end
                        break;
                    end
                end
                res(e).success = suc;
                res(e).steps   = env.steps;
                res(e).reward  = env.total_rew;
                res(e).path    = env.path;
            end

            si = find([res.success]);
            if ~isempty(si)
                [~,bi] = min([res(si).steps]);
                best_run = res(si(bi));
            else
                best_run = [];
            end

            app.drawPath(best_run, env_cfg, ROWS, COLS, NB);
            app.updateSummary(res, ag, elapsed);

            app.StatusLabel.Text = sprintf('Done! %.1fs training | %d Q-states', elapsed, ag.Q.Count);
            app.RunBtn.Enable = 'on';
        end

        %% ---- Plot helpers ----
        function plotReward(app, ag)
            ax = app.AxReward; cla(ax);
            ep = 1:numel(ag.rew);
            plot(ax, ep, ag.rew, 'Color', [app.C_ACCENT 0.12], 'LineWidth',0.5);
            hold(ax,'on');
            sm = app.smoothd(ag.rew, 300);
            if numel(sm)>1
                plot(ax, (1:numel(sm))+149, sm, 'Color',app.C_ACCENT, 'LineWidth',2);
            end
            yline(ax, 0,'--','Color',[app.C_WARN 0.5],'LineWidth',0.8);
            hold(ax,'off');
            xlabel(ax,'Episode','Color',app.C_TEXT);
            ylabel(ax,'Total Reward','Color',app.C_TEXT);
            title(ax,'Episode Reward','Color',app.C_TEXT,'FontSize',11,'FontWeight','bold');
            app.styleAx(ax);
        end

        function plotSteps(app, ag)
            ax = app.AxSteps; cla(ax);
            ep = 1:numel(ag.stp);
            plot(ax, ep, ag.stp, 'Color',[app.C_WARN 0.12],'LineWidth',0.5);
            hold(ax,'on');
            sm = app.smoothd(ag.stp, 300);
            if numel(sm)>1
                plot(ax, (1:numel(sm))+149, sm, 'Color',app.C_WARN,'LineWidth',2);
            end
            hold(ax,'off');
            xlabel(ax,'Episode','Color',app.C_TEXT);
            ylabel(ax,'Steps','Color',app.C_TEXT);
            title(ax,'Steps per Episode','Color',app.C_TEXT,'FontSize',11,'FontWeight','bold');
            app.styleAx(ax);
        end

        function plotSuccess(app, ag)
            ax = app.AxSuccess; cla(ax);
            W  = 500;
            sr = zeros(1,numel(ag.succ));
            for i=1:numel(ag.succ)
                lo=max(1,i-W+1);
                sr(i)=mean(ag.succ(lo:i))*100;
            end
            ep = 1:numel(sr);
            fill(ax,[ep fliplr(ep)],[sr zeros(1,numel(sr))], ...
                app.C_SUCCESS,'FaceAlpha',0.2,'EdgeColor','none');
            hold(ax,'on');
            plot(ax, ep, sr,'Color',app.C_SUCCESS,'LineWidth',1.6);
            hold(ax,'off');
            ylim(ax,[0 105]);
            xlabel(ax,'Episode','Color',app.C_TEXT);
            ylabel(ax,'Success Rate (%)','Color',app.C_TEXT);
            title(ax,sprintf('Rolling Success Rate  (window = %d)',W), ...
                'Color',app.C_TEXT,'FontSize',11,'FontWeight','bold');
            app.styleAx(ax);
        end

        function drawForest(app, cfg, ROWS, COLS, NB)
            ax = app.AxForest; cla(ax);
            set(ax,'Color',app.C_BG,'XColor',app.C_FIGBG,'YColor',app.C_FIGBG);
            axis(ax,'equal'); axis(ax,'off');
            xlim(ax,[-0.1 3.1]); ylim(ax,[-0.1 4.1]);
            title(ax,'Forest Layout','Color',app.C_TEXT,'FontSize',11,'FontWeight','bold');
            hold(ax,'on');
            for b=1:NB
                [row,col]=app.b2rc(b,COLS);
                y=ROWS-1-row;
                fc=app.blkClr(b,cfg);
                rectangle(ax,'Position',[col+0.05 y+0.05 0.9 0.9], ...
                    'Curvature',[0.15 0.15],'FaceColor',fc, ...
                    'EdgeColor',[0.3 0.3 0.3],'LineWidth',1.2);
                text(ax,col+0.5,y+0.58,num2str(b), ...
                    'HorizontalAlignment','center','FontSize',11, ...
                    'FontWeight','bold','Color','white');
                sym='';
                if ismember(b,cfg.r2_real), sym='[S]';
                elseif b==cfg.fake,         sym='[!]';
                elseif ismember(b,cfg.r1),  sym='R1';
                end
                if ~isempty(sym)
                    text(ax,col+0.5,y+0.22,sym,'HorizontalAlignment','center', ...
                        'FontSize',7,'Color','white');
                end
            end
            h1=scatter(ax,NaN,NaN,80,app.C_SCROLL,'s','filled','DisplayName','R2 Scroll');
            h2=scatter(ax,NaN,NaN,80,app.C_FAKE,  's','filled','DisplayName','Fake');
            h3=scatter(ax,NaN,NaN,80,app.C_R1,    's','filled','DisplayName','R1 Scroll');
            h4=scatter(ax,NaN,NaN,80,app.C_EMPTY, 's','filled','DisplayName','Empty');
            legend(ax,[h1 h2 h3 h4],'Location','southoutside','NumColumns',2, ...
                'TextColor',app.C_TEXT,'Color',app.C_FIGBG,'FontSize',8,'Box','off');
            hold(ax,'off');
        end

        function drawPath(app, best_run, cfg, ROWS, COLS, NB)
            ax = app.AxPath; cla(ax);
            set(ax,'Color',app.C_BG,'XColor',app.C_FIGBG,'YColor',app.C_FIGBG);
            axis(ax,'equal'); axis(ax,'off');
            xlim(ax,[-0.1 3.1]); ylim(ax,[-0.1 4.1]);
            title(ax,'Best Greedy Path','Color',app.C_TEXT,'FontSize',11,'FontWeight','bold');
            hold(ax,'on');
            for b=1:NB
                [row,col]=app.b2rc(b,COLS);
                y=ROWS-1-row;
                fc=app.blkClr(b,cfg);
                patch(ax,[col+0.05 col+0.95 col+0.95 col+0.05], ...
                         [y+0.05  y+0.05  y+0.95  y+0.95],fc, ...
                    'EdgeColor',[0.3 0.3 0.3],'LineWidth',1,'FaceAlpha',0.45);
                text(ax,col+0.5,y+0.5,num2str(b),'HorizontalAlignment','center', ...
                    'FontSize',11,'FontWeight','bold','Color','white');
            end
            if ~isempty(best_run)
                blks=[];
                for i=1:numel(best_run.path)
                    s=best_run.path{i};
                    if isnumeric(s), blks(end+1)=s; end %#ok<AGROW>
                end
                if numel(blks)>=2
                    px=zeros(1,numel(blks)); py=zeros(1,numel(blks));
                    for i=1:numel(blks)
                        [ri,ci]=app.b2rc(blks(i),COLS);
                        px(i)=ci+0.5; py(i)=(ROWS-1-ri)+0.5;
                    end
                    plot(ax,px,py,'-','Color',app.C_ACCENT,'LineWidth',2.8);
                    % Arrowhead on last segment
                    quiver(ax,px(end-1),py(end-1), ...
                           px(end)-px(end-1),py(end)-py(end-1),0, ...
                           'Color',app.C_ACCENT,'LineWidth',2.8,'MaxHeadSize',1.8);
                    % Start marker
                    scatter(ax,px(1),py(1),150,app.C_SUCCESS,'o','filled', ...
                        'MarkerEdgeColor','white','LineWidth',1.5);
                    text(ax,px(1)+0.08,py(1)+0.12,'START', ...
                        'Color',app.C_SUCCESS,'FontSize',7,'FontWeight','bold');
                end
            else
                text(ax,1.5,2,'No successful path found', ...
                    'HorizontalAlignment','center','Color',app.C_WARN, ...
                    'FontSize',11,'FontWeight','bold');
            end
            hold(ax,'off');
        end

        function updateSummary(app, res, ag, elapsed)
            si = find([res.success]);
            n  = numel(si);
            sr = n/numel(res)*100;
            if n>0
                as_v=mean([res(si).steps]);
                ms_v=min( [res(si).steps]);
                ar_v=mean([res(si).reward]);
            else
                as_v=0; ms_v=0; ar_v=0;
            end
            app.SRVal.Text = sprintf('%.1f%%', sr);
            app.ASVal.Text = sprintf('%.1f',   as_v);
            app.MSVal.Text = sprintf('%d',      ms_v);
            app.ARVal.Text = sprintf('%.1f',   ar_v);
            app.EPVal.Text = sprintf('%d',      numel(res));
            app.QSVal.Text = sprintf('%d',      ag.Q.Count);
            app.BTVal.Text = sprintf('%.2f s',  elapsed);
            if sr>=90,     app.SRVal.FontColor=app.C_SUCCESS;
            elseif sr>=50, app.SRVal.FontColor=app.C_WARN;
            else,          app.SRVal.FontColor=app.C_DANGER;
            end
        end

        function styleAx(app, ax)
            set(ax,'Color',app.C_BG,'XColor',app.C_TEXT,'YColor',app.C_TEXT, ...
                'GridColor',app.C_GRID,'GridAlpha',0.6,'GridLineStyle','--','FontSize',9);
            grid(ax,'on'); box(ax,'off');
        end

        function fc = blkClr(app, b, cfg)
            if ismember(b,cfg.r2_real), fc=app.C_SCROLL;
            elseif b==cfg.fake,         fc=app.C_FAKE;
            elseif ismember(b,cfg.r1),  fc=app.C_R1;
            else,                        fc=app.C_EMPTY;
            end
        end

        %% ---- Smooth helper ----
        function s = smoothd(~, data, W)
            if numel(data)<W, s=data; return; end
            s = conv(data, ones(1,W)/W, 'valid');
        end

        %% ---- Grid helpers ----
        function [r,c] = b2rc(~, b, COLS)
            r = floor((b-1)/COLS);
            c = mod(b-1, COLS);
        end
        function b = rc2b(~, r, c, COLS)
            b = r*COLS + c + 1;
        end

        %% ---- Environment ----
        function env = envReset(app, env)
            e = intersect(env.ENTRY, env.r2_real);
            if ~isempty(e), env.current = e(randi(numel(e)));
            else,           env.current = env.ENTRY(randi(numel(env.ENTRY)));
            end
            env.collected   = [];
            env.remaining   = env.r2_real;
            env.done        = false;
            env.steps       = 0;
            env.total_rew   = 0;
            env.path        = {env.current};
            env.vis         = zeros(1,env.NB);
            env.vis(env.current)=1;
        end

        function st = getState(~, env)
            col_s = sprintf('%d,',sort(env.collected));
            rem_s = sprintf('%d,',sort(env.remaining));
            r1_s  = sprintf('%d,',sort(env.r1));
            st = sprintf('b%d|c%s|r%s|f%d|r1%s', ...
                env.current,col_s,rem_s,env.fake,r1_s);
        end

        function acts = legalActs(app, env)
            acts=[];
            [r,c]=app.b2rc(env.current,env.COLS);
            if r>0,           acts(end+1)=env.UP;    end
            if r<env.ROWS-1,  acts(end+1)=env.DOWN;  end
            if c>0,           acts(end+1)=env.LEFT;  end
            if c<env.COLS-1,  acts(end+1)=env.RIGHT; end
            nb=union(env.current, env.ADJ{env.current});
            if ~isempty(intersect(nb,env.remaining))
                acts(end+1)=env.COLLECT;
            end
            if isempty(env.remaining) && ismember(env.current,env.EXIT)
                acts(end+1)=env.EXIT_A;
            end
        end

        function [env,reward,done,info] = step(app, env, action)
            reward=0; done=false; info=struct();
            [r,c]=app.b2rc(env.current,env.COLS);

            if ismember(action,[env.UP env.DOWN env.LEFT env.RIGHT])
                switch action
                    case env.UP,    dr=-1;dc=0;
                    case env.DOWN,  dr= 1;dc=0;
                    case env.LEFT,  dr= 0;dc=-1;
                    otherwise,      dr= 0;dc= 1;
                end
                nr=r+dr; nc=c+dc;
                if nr>=0&&nr<env.ROWS&&nc>=0&&nc<env.COLS
                    nb=app.rc2b(nr,nc,env.COLS);
                    if nb==env.fake, reward=reward+env.R_FAKEPROX; end
                    env.current=nb;
                    env.steps=env.steps+1;
                    env.vis(nb)=env.vis(nb)+1;
                    env.path{end+1}=nb;
                    if env.vis(nb)>2 && ~ismember(nb,env.remaining)
                        reward=reward+env.R_WANDER;
                    end
                    reward=reward+env.R_MOVE;
                    if isempty(env.collected), reward=reward+env.R_EXTRA*0.5; end
                    info.moved_to=nb;
                else
                    reward=reward+env.R_INV; info.error='wall';
                end

            elseif action==env.COLLECT
                nb=union(env.current, env.ADJ{env.current});
                if ismember(env.fake,nb) && env.fake==env.current
                    reward=reward+env.R_FAKE; done=true; info.error='fake';
                elseif ~isempty(intersect(env.r1,nb)) && isempty(intersect(env.remaining,nb))
                    reward=reward+env.R_R1; info.error='r1';
                elseif ~isempty(intersect(env.remaining,nb))
                    coll=intersect(env.remaining,nb);
                    if ismember(env.current,coll), tgt=env.current;
                    else, tgt=coll(1); end
                    env.collected=union(env.collected,tgt);
                    env.remaining=setdiff(env.remaining,tgt);
                    reward=reward+env.R_REAL;
                    if isempty(env.remaining), reward=reward+env.R_ALL; end
                    env.steps=env.steps+1;
                    env.path{end+1}=sprintf('C%d',tgt);
                else
                    reward=reward+env.R_INV;
                end

            elseif action==env.EXIT_A
                if isempty(env.remaining)&&ismember(env.current,env.EXIT)
                    reward=reward+env.R_EXIT; done=true;
                    info.success=true; env.path{end+1}='EXIT';
                else
                    reward=reward+env.R_INV;
                end
            else
                reward=reward+env.R_INV;
            end
            env.done=done; env.total_rew=env.total_rew+reward;
        end

        %% ---- Agent ----
        function act = selectAct(app, ag, st, legal, greedy)
            if ~greedy && rand()<ag.eps
                act=legal(randi(numel(legal))); return;
            end
            bq=-inf; ba=legal(1);
            for i=1:numel(legal)
                q=app.getQ(ag,st,legal(i));
                if q>bq, bq=q; ba=legal(i); end
            end
            act=ba;
        end

        function q = getQ(~, ag, st, act)
            k=sprintf('%s|a%d',st,act);
            if ag.Q.isKey(k), q=ag.Q(k); else, q=ag.Q_opt; end
        end

        function ag = updateQ(app, ag, st, act, r, nst, nlegal, done)
            k=sprintf('%s|a%d',st,act);
            if ag.visits.isKey(k), n=ag.visits(k)+1; else, n=1; end
            ag.visits(k)=n;
            alpha_n=ag.alpha/(1+n/200);
            if done
                tgt=r;
            else
                mq=-inf;
                for i=1:numel(nlegal)
                    q=app.getQ(ag,nst,nlegal(i));
                    if q>mq, mq=q; end
                end
                if mq==-inf, mq=0; end
                tgt=r+ag.gamma*mq;
            end
            old=app.getQ(ag,st,act);
            ag.Q(k)=old+alpha_n*(tgt-old);
        end

    end % private methods

    %% ---- Component Initialisation ----
    methods (Access = private)

        function createComponents(app)
            BG   = app.C_BG;
            FBG  = app.C_FIGBG;
            TC   = app.C_TEXT;
            ACC  = app.C_ACCENT;

            %% Figure
            app.UIFigure = uifigure('Visible','off');
            app.UIFigure.Color = FBG;
            app.UIFigure.Position = [80 60 1440 860];
            app.UIFigure.Name = 'ABU Robocon 2026 - R2 Q-Learning Visualizer';

            %% Left config panel
            app.LeftPanel = uipanel(app.UIFigure, ...
                'Position',[10 10 220 840], ...
                'BackgroundColor',BG,'ForegroundColor',TC, ...
                'Title','','BorderType','none');

            app.TitleLabel = uilabel(app.LeftPanel, ...
                'Text','R2 Q-Learning', ...
                'Position',[5 795 210 36],'FontSize',14, ...
                'FontWeight','bold','FontColor',ACC,'BackgroundColor',BG);

            % Episodes
            uilabel(app.LeftPanel,'Text','Episodes', ...
                'Position',[5 745 150 20],'FontColor',TC,'BackgroundColor',BG,'FontSize',10);
            app.EpSpinner = uispinner(app.LeftPanel,'Value',10000, ...
                'Limits',[100 100000],'Step',1000, ...
                'Position',[5 720 180 26],'FontColor',TC,'BackgroundColor',BG);

            % Max Steps
            uilabel(app.LeftPanel,'Text','Max Steps / Episode', ...
                'Position',[5 690 200 20],'FontColor',TC,'BackgroundColor',BG,'FontSize',10);
            app.StepsSpinner = uispinner(app.LeftPanel,'Value',200, ...
                'Limits',[20 1000],'Step',10, ...
                'Position',[5 665 180 26],'FontColor',TC,'BackgroundColor',BG);

            % Alpha
            uilabel(app.LeftPanel,'Text','Learning Rate (alpha)', ...
                'Position',[5 635 200 20],'FontColor',TC,'BackgroundColor',BG,'FontSize',10);
            app.AlphaSpinner = uispinner(app.LeftPanel,'Value',0.15, ...
                'Limits',[0.001 1],'Step',0.01, ...
                'Position',[5 610 180 26],'FontColor',TC,'BackgroundColor',BG);

            % Gamma
            uilabel(app.LeftPanel,'Text','Discount Factor (gamma)', ...
                'Position',[5 580 200 20],'FontColor',TC,'BackgroundColor',BG,'FontSize',10);
            app.GammaSpinner = uispinner(app.LeftPanel,'Value',0.97, ...
                'Limits',[0.1 0.9999],'Step',0.01, ...
                'Position',[5 555 180 26],'FontColor',TC,'BackgroundColor',BG);

            % R2 Scrolls
            uilabel(app.LeftPanel,'Text','R2 Scrolls (space-separated)', ...
                'Position',[5 520 210 20],'FontColor',TC,'BackgroundColor',BG,'FontSize',10);
            app.R2Field = uieditfield(app.LeftPanel,'text','Value','1 7 5 11', ...
                'Position',[5 495 180 26],'FontColor',TC,'BackgroundColor',BG);

            % Fake Scroll
            uilabel(app.LeftPanel,'Text','Fake Scroll Block #', ...
                'Position',[5 465 200 20],'FontColor',TC,'BackgroundColor',BG,'FontSize',10);
            app.FakeSpinner = uispinner(app.LeftPanel,'Value',4, ...
                'Limits',[1 12],'Step',1, ...
                'Position',[5 440 180 26],'FontColor',TC,'BackgroundColor',BG);

            % Divider note
            uilabel(app.LeftPanel,'Text','R1 Scrolls fixed: [6,10,12]', ...
                'Position',[5 408 210 18],'FontColor',[0.5 0.5 0.5], ...
                'BackgroundColor',BG,'FontSize',9);

            % Run button
            app.RunBtn = uibutton(app.LeftPanel,'push', ...
                'Text','Run Training', ...
                'Position',[5 360 180 38], ...
                'FontSize',12,'FontWeight','bold', ...
                'FontColor','black','BackgroundColor',ACC, ...
                'ButtonPushedFcn',@(~,~)app.RunButtonPushed([]));

            % Status
            app.StatusLabel = uilabel(app.LeftPanel, ...
                'Text','Configure and press Run', ...
                'Position',[5 320 210 34],'FontSize',9, ...
                'FontColor',TC,'BackgroundColor',BG,'WordWrap','on');

            %% Tab group
            app.TabGroup = uitabgroup(app.UIFigure,'Position',[240 10 1190 840]);

            %% Tab 1 - Reward
            app.Tab1 = uitab(app.TabGroup,'Title','Reward Curve');
            app.Tab1.BackgroundColor = FBG;
            app.AxReward = uiaxes(app.Tab1,'Position',[20 20 1140 790]);
            app.AxReward.Color = BG;
            app.styleAx(app.AxReward);

            %% Tab 2 - Steps
            app.Tab2 = uitab(app.TabGroup,'Title','Steps per Episode');
            app.Tab2.BackgroundColor = FBG;
            app.AxSteps = uiaxes(app.Tab2,'Position',[20 20 1140 790]);
            app.AxSteps.Color = BG;
            app.styleAx(app.AxSteps);

            %% Tab 3 - Success Rate
            app.Tab3 = uitab(app.TabGroup,'Title','Success Rate');
            app.Tab3.BackgroundColor = FBG;
            app.AxSuccess = uiaxes(app.Tab3,'Position',[20 20 1140 790]);
            app.AxSuccess.Color = BG;
            app.styleAx(app.AxSuccess);

            %% Tab 4 - Forest Layout
            app.Tab4 = uitab(app.TabGroup,'Title','Forest Layout');
            app.Tab4.BackgroundColor = FBG;
            app.AxForest = uiaxes(app.Tab4,'Position',[170 30 820 760]);
            app.AxForest.Color = BG;

            %% Tab 5 - Best Path
            app.Tab5 = uitab(app.TabGroup,'Title','Best Path');
            app.Tab5.BackgroundColor = FBG;
            app.AxPath = uiaxes(app.Tab5,'Position',[170 30 820 760]);
            app.AxPath.Color = BG;

            %% Tab 6 - Summary
            app.Tab6 = uitab(app.TabGroup,'Title','Summary');
            app.Tab6.BackgroundColor = FBG;

            app.SumPanel = uipanel(app.Tab6, ...
                'Title','Evaluation Results','Position',[320 180 520 520], ...
                'BackgroundColor',BG,'ForegroundColor',TC,'FontSize',12);

            rows = {'Success Rate','Avg Steps','Min Steps','Avg Reward', ...
                    'Eval Episodes','Q-Table States','Training Time'};
            tFields = {'SRTitle','ASTitle','MSTitle','ARTitle','EPTitle','QSTitle','BTTitle'};
            vFields = {'SRVal',  'ASVal',  'MSVal',  'ARVal',  'EPVal',  'QSVal',  'BTVal'};
            vColors = {app.C_SUCCESS, app.C_ACCENT, app.C_SUCCESS, ...
                       app.C_WARN, TC, TC, TC};

            for i = 1:numel(rows)
                yp = 460 - (i-1)*60;
                app.(tFields{i}) = uilabel(app.SumPanel, ...
                    'Text',rows{i},'Position',[20 yp 200 30], ...
                    'FontSize',13,'FontColor',TC,'BackgroundColor',BG);
                app.(vFields{i}) = uilabel(app.SumPanel, ...
                    'Text','—','Position',[280 yp 200 30], ...
                    'FontSize',14,'FontWeight','bold', ...
                    'FontColor',vColors{i},'BackgroundColor',BG);
            end

            app.UIFigure.Visible = 'on';
        end

    end % private createComponents

    %% ---- App lifecycle ----
    methods (Access = public)
        function app = RoboR2
            createComponents(app);
            registerApp(app, app.UIFigure);
            if nargout == 0, clear app; end
        end
        function delete(app)
            delete(app.UIFigure);
        end
    end

end
