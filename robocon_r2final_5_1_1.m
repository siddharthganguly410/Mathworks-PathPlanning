%% =========================================================
%  ABU Robocon 2026 – "Kung Fu Quest"
%  Robot 2 (R2) – Meihua Forest Q-Learning Solver
% =========================================================
%  Grid layout (1-indexed):
%      1   2   3
%      4   5   6
%      7   8   9
%     10  11  12
%
%  R2 enters from {1,2,3}, exits through {10,11,12}.
% =========================================================

clc; clear; close all;
rng(42);   % reproducible

%% ─────────────────────────────────────────────
%   CONSTANTS
%% ─────────────────────────────────────────────
ROWS = 4;  COLS = 3;  NUM_BLOCKS = 12;

% Actions (1-indexed to avoid confusion with MATLAB indexing)
UP      = 1;
DOWN    = 2;
LEFT    = 3;
RIGHT   = 4;
COLLECT = 5;
EXIT_A  = 6;
ACTION_NAMES = {"UP","DOWN","LEFT","RIGHT","COLLECT","EXIT"};

% Rewards
R_COLLECT_REAL   =  100;
R_COLLECT_ALL    =  500;
R_EXIT_SUCCESS   =  300;
R_MOVE           =   -1;
R_EXTRA_MOVE     =   -2;
R_FAKE_PROXIMITY = -500;
R_COLLECT_FAKE   = -1000;
R_COLLECT_R1     = -1000;
R_INVALID        = -500;
R_WANDER         =   -5;

ENTRY_BLOCKS = [1, 2, 3];
EXIT_BLOCKS  = [10, 11, 12];

% Build adjacency list
ADJACENCY = cell(NUM_BLOCKS, 1);
for b = 1:NUM_BLOCKS
    [r, c] = block_to_rc(b, COLS);
    nbrs = [];
    if r > 0,          nbrs(end+1) = rc_to_block(r-1, c, COLS); end  % UP
    if r < ROWS-1,     nbrs(end+1) = rc_to_block(r+1, c, COLS); end  % DOWN
    if c > 0,          nbrs(end+1) = rc_to_block(r, c-1, COLS); end  % LEFT
    if c < COLS-1,     nbrs(end+1) = rc_to_block(r, c+1, COLS); end  % RIGHT
    ADJACENCY{b} = nbrs;
end

%% ─────────────────────────────────────────────
%   CONFIGURATION
%% ─────────────────────────────────────────────
% ── Read config from GUI if available, else use defaults ──
if evalin('base','exist(''gui_r2_real_scrolls'',''var'')')
    r2_real_scrolls = evalin('base','gui_r2_real_scrolls');
else
    r2_real_scrolls = [1, 7, 5, 11];
end

if evalin('base','exist(''gui_fake_scroll'',''var'')')
    fake_scroll = evalin('base','gui_fake_scroll');
else
    fake_scroll = 4;
end

if evalin('base','exist(''gui_r1_scrolls'',''var'')')
    r1_scrolls = evalin('base','gui_r1_scrolls');
else
    r1_scrolls = [12, 6, 10];
end

env_config.r2_real = r2_real_scrolls;
env_config.fake    = fake_scroll;
env_config.r1      = r1_scrolls;

fprintf('\n  Forest Config:\n');
fprintf('    R2 Real Scrolls : [%s]\n', num2str(r2_real_scrolls));
fprintf('    Fake Scroll     : %d\n',   fake_scroll);
fprintf('    R1 Scrolls      : [%s]\n', num2str(r1_scrolls));

%% ─────────────────────────────────────────────
%   ENVIRONMENT STRUCT
%% ─────────────────────────────────────────────
env.r2_real_scrolls = r2_real_scrolls;
env.fake_scroll     = fake_scroll;
env.r1_scrolls      = r1_scrolls;
env.ROWS = ROWS; env.COLS = COLS; env.NUM_BLOCKS = NUM_BLOCKS;
env.ADJACENCY = ADJACENCY;
env.ENTRY_BLOCKS = ENTRY_BLOCKS;
env.EXIT_BLOCKS  = EXIT_BLOCKS;
% Reward constants bundled into env for passing to functions
env.R_COLLECT_REAL   = R_COLLECT_REAL;
env.R_COLLECT_ALL    = R_COLLECT_ALL;
env.R_EXIT_SUCCESS   = R_EXIT_SUCCESS;
env.R_MOVE           = R_MOVE;
env.R_EXTRA_MOVE     = R_EXTRA_MOVE;
env.R_FAKE_PROXIMITY = R_FAKE_PROXIMITY;
env.R_COLLECT_FAKE   = R_COLLECT_FAKE;
env.R_COLLECT_R1     = R_COLLECT_R1;
env.R_INVALID        = R_INVALID;
env.R_WANDER         = R_WANDER;
env.UP=UP; env.DOWN=DOWN; env.LEFT=LEFT; env.RIGHT=RIGHT;
env.COLLECT=COLLECT; env.EXIT_A=EXIT_A;

env = env_reset(env);

%% ─────────────────────────────────────────────
%   Q-LEARNING AGENT
%% ─────────────────────────────────────────────
agent.alpha          = 0.15;
agent.gamma          = 0.97;
agent.epsilon        = 1.0;
agent.epsilon_min    = 0.01;
agent.epsilon_decay  = 0.9997;
agent.optimistic_init = 10.0;
agent.q_table        = containers.Map('KeyType','char','ValueType','any');
agent.visit_counts   = containers.Map('KeyType','char','ValueType','any');
agent.episode_rewards  = [];
agent.episode_steps    = [];
agent.episode_success  = [];

%% ─────────────────────────────────────────────
%   TRAIN
%% ─────────────────────────────────────────────
EPISODES   = 10000;
MAX_STEPS  = 200;
VERBOSE_EP = 2000;
PATIENCE   = 3000;

fprintf('\n%s\n', repmat('=',1,56));
fprintf('  ABU Robocon 2026 – R2 Q-Learning Training\n');
fprintf('  Episodes: %d   Max steps/ep: %d\n', EPISODES, MAX_STEPS);
fprintf('%s\n\n', repmat('=',1,56));

t0 = tic;
best_reward    = -inf;
best_path      = {};
success_window = zeros(1, PATIENCE);
sw_ptr         = 0;          % circular-buffer pointer
sw_count       = 0;

for ep = 1:EPISODES
    env   = env_reset(env);
    state = env_get_state(env);
    ep_rew  = 0.0;
    success = false;

    for stp = 1:MAX_STEPS
        legal = env_legal_actions(env);
        if isempty(legal), break; end

        action = agent_select_action(agent, state, legal, false);
        [env, reward, done, info] = env_step(env, action);
        next_state  = env_get_state(env);
        next_legal  = env_legal_actions(env);

        agent = agent_update(agent, state, action, reward, ...
                             next_state, next_legal, done);
        state  = next_state;
        ep_rew = ep_rew + reward;

        if done
            if isfield(info,'success') && info.success
                success = true;
            end
            break;
        end
    end

    agent.epsilon        = max(agent.epsilon_min, ...
                               agent.epsilon * agent.epsilon_decay);
    agent.episode_rewards(end+1) = ep_rew;
    agent.episode_steps(end+1)   = env.steps;
    agent.episode_success(end+1) = double(success);

    % Circular success window
    sw_ptr = mod(sw_ptr, PATIENCE) + 1;
    success_window(sw_ptr) = double(success);
    if sw_count < PATIENCE, sw_count = sw_count + 1; end

    if ep_rew > best_reward && success
        best_reward = ep_rew;
        best_path   = env.path;
    end

    if mod(ep, VERBOSE_EP) == 0
        recent_n  = min(VERBOSE_EP, ep);
        avg_rew   = mean(agent.episode_rewards(end-recent_n+1:end));
        avg_steps = mean(agent.episode_steps(end-recent_n+1:end));
        succ_rate = mean(agent.episode_success(end-recent_n+1:end))*100;
        fprintf('  Ep %6d/%d  eps=%.4f  AvgRew=%8.1f  AvgSteps=%5.1f  SuccessRate=%5.1f%%\n', ...
            ep, EPISODES, agent.epsilon, avg_rew, avg_steps, succ_rate);
    end

    % Early stopping
    if sw_count == PATIENCE && sum(success_window)/PATIENCE >= 0.98
        fprintf('\n  Early-stop at episode %d: success rate >= 98%% over last %d episodes\n', ...
            ep, PATIENCE);
        break;
    end
end

elapsed = toc(t0);
fprintf('\n  Best episode reward : %.1f\n', best_reward);
fprintf('  Training time       : %.1fs\n', elapsed);
fprintf('  Q-table states      : %d\n\n', agent.q_table.Count);

%% ─────────────────────────────────────────────
%   EVALUATE – GREEDY ROLLOUTS
%% ─────────────────────────────────────────────
EVAL_EPISODES = 200;
EVAL_MAXSTEPS = 100;

eval_results = struct('success',{},'steps',{},'reward',{},'path',{});

for e = 1:EVAL_EPISODES
    env   = env_reset(env);
    state = env_get_state(env);
    suc   = false;
    for stp = 1:EVAL_MAXSTEPS
        legal = env_legal_actions(env);
        if isempty(legal), break; end
        action = agent_select_action(agent, state, legal, true);
        [env, ~, done, info] = env_step(env, action);
        state = env_get_state(env);
        if done
            if isfield(info,'success') && info.success
                suc = true;
            end
            break;
        end
    end
    eval_results(e).success = suc;
    eval_results(e).steps   = env.steps;
    eval_results(e).reward  = env.total_reward;
    eval_results(e).path    = env.path;
end

succ_idx   = find([eval_results.success]);
succ_rate  = numel(succ_idx)/EVAL_EPISODES*100;
if ~isempty(succ_idx)
    avg_steps  = mean([eval_results(succ_idx).steps]);
    avg_reward = mean([eval_results(succ_idx).reward]);
    [~,bi]     = min([eval_results(succ_idx).steps]);
    best_run   = eval_results(succ_idx(bi));
else
    avg_steps  = NaN;  avg_reward = NaN;  best_run = [];
end

fprintf('%s\n', repmat('-',1,50));
fprintf('  Evaluation over %d greedy episodes:\n', EVAL_EPISODES);
fprintf('    Success rate  : %.1f%%\n', succ_rate);
fprintf('    Avg steps     : %.1f\n',   avg_steps);
fprintf('    Avg reward    : %.1f\n',   avg_reward);
if ~isempty(best_run)
    fprintf('    Best steps    : %d\n', best_run.steps);
end
fprintf('%s\n\n', repmat('-',1,50));

%% ─────────────────────────────────────────────
%   PRINT OPTIMAL PATH
%% ─────────────────────────────────────────────
print_optimal_path(eval_results, env_config, COLS);

%% ─────────────────────────────────────────────
%   PLOTTING
%% ─────────────────────────────────────────────
plot_training(agent, eval_results, env_config, best_run, ROWS, COLS, NUM_BLOCKS);

%% ==========================================================
%   LOCAL HELPER FUNCTIONS
%% ==========================================================

% ── grid helpers ──────────────────────────────────────────
function [r, c] = block_to_rc(b, COLS)
    r = floor((b-1)/COLS);
    c = mod(b-1, COLS);
end

function b = rc_to_block(r, c, COLS)
    b = r * COLS + c + 1;
end

%% ─────────────────────────────────────────────
%   ENVIRONMENT FUNCTIONS
%% ─────────────────────────────────────────────
function env = env_reset(env)
    entry_with_scroll = intersect(env.ENTRY_BLOCKS, env.r2_real_scrolls);
    if ~isempty(entry_with_scroll)
        idx = randi(numel(entry_with_scroll));
        env.current_block = entry_with_scroll(idx);
    else
        idx = randi(numel(env.ENTRY_BLOCKS));
        env.current_block = env.ENTRY_BLOCKS(idx);
    end
    env.collected     = [];
    env.remaining     = env.r2_real_scrolls;
    env.done          = false;
    env.steps         = 0;
    env.total_reward  = 0.0;
    env.path          = {env.current_block};
    env.visit_count   = zeros(1, env.NUM_BLOCKS);
    env.visit_count(env.current_block) = 1;
end

function state = env_get_state(env)
    % Encode state as a compact string key for the Q-table map
    col_str  = sprintf('%d,', sort(env.collected));
    rem_str  = sprintf('%d,', sort(env.remaining));
    r1_str   = sprintf('%d,', sort(env.r1_scrolls));
    state = sprintf('b%d|c%s|r%s|f%d|r1%s', ...
        env.current_block, col_str, rem_str, env.fake_scroll, r1_str);
end

function acts = env_legal_actions(env)
    acts = [];
    [r, c] = block_to_rc(env.current_block, env.COLS);
    if r > 0,            acts(end+1) = env.UP;    end
    if r < env.ROWS-1,   acts(end+1) = env.DOWN;  end
    if c > 0,            acts(end+1) = env.LEFT;  end
    if c < env.COLS-1,   acts(end+1) = env.RIGHT; end

    % COLLECT: legal if current block or neighbour has remaining scroll
    nearby = union(env.current_block, env.ADJACENCY{env.current_block});
    if ~isempty(intersect(nearby, env.remaining))
        acts(end+1) = env.COLLECT;
    end

    % EXIT: all collected and in exit zone
    if isempty(env.remaining) && ismember(env.current_block, env.EXIT_BLOCKS)
        acts(end+1) = env.EXIT_A;
    end
end

function [env, reward, done, info] = env_step(env, action)
    reward = 0.0;
    done   = false;
    info   = struct();

    [r, c] = block_to_rc(env.current_block, env.COLS);

    if ismember(action, [env.UP, env.DOWN, env.LEFT, env.RIGHT])
        dr_map = [env.UP, -1; env.DOWN, 1; env.LEFT, 0;  env.RIGHT, 0];
        dc_map = [env.UP,  0; env.DOWN, 0; env.LEFT, -1; env.RIGHT, 1];
        dr = dr_map(dr_map(:,1)==action, 2);
        dc = dc_map(dc_map(:,1)==action, 2);
        nr = r + dr;  nc = c + dc;

        if nr >= 0 && nr < env.ROWS && nc >= 0 && nc < env.COLS
            new_block = rc_to_block(nr, nc, env.COLS);

            if new_block == env.fake_scroll
                reward = reward + env.R_FAKE_PROXIMITY;
            end

            env.current_block = new_block;
            env.steps         = env.steps + 1;
            env.visit_count(new_block) = env.visit_count(new_block) + 1;
            env.path{end+1}   = new_block;

            if env.visit_count(new_block) > 2 && ~ismember(new_block, env.remaining)
                reward = reward + env.R_WANDER;
            end

            reward = reward + env.R_MOVE;
            if isempty(env.collected)
                reward = reward + env.R_EXTRA_MOVE * 0.5;
            end
            info.moved_to = new_block;
        else
            reward = reward + env.R_INVALID;
            info.error = 'wall';
        end

    elseif action == env.COLLECT
        nearby = union(env.current_block, env.ADJACENCY{env.current_block});

        if ismember(env.fake_scroll, nearby) && env.fake_scroll == env.current_block
            reward = reward + env.R_COLLECT_FAKE;
            done   = true;
            info.error = 'collected_fake';

        elseif ~isempty(intersect(env.r1_scrolls, nearby)) && ...
               isempty(intersect(env.remaining, nearby))
            reward = reward + env.R_COLLECT_R1;
            info.error = 'attempted_r1';

        elseif ~isempty(intersect(env.remaining, nearby))
            collectible = intersect(env.remaining, nearby);
            if ismember(env.current_block, collectible)
                target = env.current_block;
            else
                target = collectible(1);
            end
            env.collected = union(env.collected, target);
            env.remaining = setdiff(env.remaining, target);
            reward = reward + env.R_COLLECT_REAL;
            info.collected = target;

            if isempty(env.remaining)
                reward = reward + env.R_COLLECT_ALL;
                info.all_collected = true;
            end
            env.steps     = env.steps + 1;
            env.path{end+1} = sprintf('C%d', target);
        else
            reward = reward + env.R_INVALID;
            info.error = 'nothing_to_collect';
        end

    elseif action == env.EXIT_A
        if isempty(env.remaining) && ismember(env.current_block, env.EXIT_BLOCKS)
            reward = reward + env.R_EXIT_SUCCESS;
            done   = true;
            info.success = true;
            env.path{end+1} = 'EXIT';
        else
            reward = reward + env.R_INVALID;
            info.error = 'invalid_exit';
        end
    else
        reward = reward + env.R_INVALID;
        info.error = 'unknown_action';
    end

    env.done         = done;
    env.total_reward = env.total_reward + reward;
end

%% ─────────────────────────────────────────────
%   AGENT FUNCTIONS
%% ─────────────────────────────────────────────
function action = agent_select_action(agent, state, legal_actions, greedy)
    if ~greedy && rand() < agent.epsilon
        action = legal_actions(randi(numel(legal_actions)));
        return;
    end

    best_q = -inf;
    best_a = legal_actions(1);
    for i = 1:numel(legal_actions)
        a  = legal_actions(i);
        qv = agent_get_q(agent, state, a);
        if qv > best_q
            best_q = qv;
            best_a = a;
        end
    end
    action = best_a;
end

function q = agent_get_q(agent, state, action)
    key = sprintf('%s|a%d', state, action);
    if agent.q_table.isKey(key)
        q = agent.q_table(key);
    else
        q = agent.optimistic_init;
    end
end

function agent = agent_update(agent, state, action, reward, ...
                               next_state, next_legal, done)
    sa_key = sprintf('%s|a%d', state, action);

    % Visit count for harmonic LR
    if agent.visit_counts.isKey(sa_key)
        n = agent.visit_counts(sa_key) + 1;
    else
        n = 1;
    end
    agent.visit_counts(sa_key) = n;
    alpha_n = agent.alpha / (1.0 + n/200.0);

    if done
        target = reward;
    else
        max_next_q = -inf;
        for i = 1:numel(next_legal)
            qv = agent_get_q(agent, next_state, next_legal(i));
            if qv > max_next_q, max_next_q = qv; end
        end
        if max_next_q == -inf, max_next_q = 0; end
        target = reward + agent.gamma * max_next_q;
    end

    old_q = agent_get_q(agent, state, action);
    agent.q_table(sa_key) = old_q + alpha_n * (target - old_q);
end

%% ─────────────────────────────────────────────
%   PRINT OPTIMAL PATH
%% ─────────────────────────────────────────────
function print_optimal_path(eval_results, env_config, COLS)
    succ_idx = find([eval_results.success]);
    if isempty(succ_idx)
        fprintf('  No successful paths found in evaluation.\n\n');
        return;
    end
    [~,bi]  = min([eval_results(succ_idx).steps]);
    best    = eval_results(succ_idx(bi));

    fprintf('\n%s\n', repmat('=',1,56));
    fprintf('  OPTIMAL PATH  (steps=%d, reward=%.1f)\n', best.steps, best.reward);
    fprintf('%s\n', repmat('=',1,56));
    fprintf('  R2 Scrolls : [%s]\n', num2str(sort(env_config.r2_real)));
    fprintf('  Fake Scroll: %d\n',   env_config.fake);
    fprintf('  R1 Scrolls : [%s]\n', num2str(sort(env_config.r1)));

    prev_block = -1;
    for i = 1:numel(best.path)
        step = best.path{i};
        if isnumeric(step)
            direction = '';
            if prev_block > 0
                [pr,pc] = block_to_rc(prev_block, COLS);
                [cr,cc] = block_to_rc(step, COLS);
                dr = cr-pr;  dc = cc-pc;
                if dr==-1 && dc==0, direction = 'UP   ';
                elseif dr==1 && dc==0, direction = 'DOWN ';
                elseif dr==0 && dc==-1, direction = 'LEFT ';
                elseif dr==0 && dc==1,  direction = 'RIGHT';
                end
            end
            tag = '';
            if ismember(step, env_config.r2_real), tag = ' [R2 Scroll]';
            elseif step == env_config.fake,         tag = ' [FAKE]';
            elseif ismember(step, env_config.r1),   tag = ' [R1]';
            end
            fprintf('  %3d. Block %2d  %-10s%s\n', i, step, direction, tag);
            prev_block = step;
        elseif ischar(step) && step(1) == 'C'
            cb = str2double(step(2:end));
            fprintf('       -> COLLECT scroll from block %d\n', cb);
        elseif strcmp(step,'EXIT')
            fprintf('       -> EXIT\n');
        end
    end
    fprintf('%s\n\n', repmat('=',1,56));
end

%% ─────────────────────────────────────────────
%   SMOOTH (moving average)
%% ─────────────────────────────────────────────
function s = smooth_data(data, window)
    n = numel(data);
    if n < window, s = data; return; end
    kernel = ones(1,window)/window;
    s = conv(data, kernel, 'valid');
end

%% ─────────────────────────────────────────────
%   PLOTTING
%% ─────────────────────────────────────────────
function plot_training(agent, eval_results, env_cfg, best_run, ROWS, COLS, NUM_BLOCKS)

    % ── Colour palette ──
    ACCENT   = [0.000, 0.898, 1.000];   % #00e5ff  - cyan
    BG       = [0.102, 0.114, 0.180];   % #1a1d2e
    FIGBG    = [0.059, 0.067, 0.090];   % #0f1117
    TEXT     = [0.878, 0.878, 0.878];   % #e0e0e0
    SCROLL_C = [1.000, 0.843, 0.000];   % #ffd700  - gold
    FAKE_C   = [1.000, 0.302, 0.427];   % #ff4d6d  - red
    R1_C     = [0.706, 0.557, 0.678];   % #b48ead  - mauve
    EMPTY_C  = [0.165, 0.176, 0.243];   % #2a2d3e  - dark

    % ── Single fullscreen figure ──
    fig = figure('Color', FIGBG, ...
                 'Position', [100 80 800 920], ...
                 'Name', 'Best Greedy Path');

    ax = axes('Parent', fig, ...
              'Color', BG, ...
              'XColor', FIGBG, 'YColor', FIGBG, ...
              'Position', [0.05 0.05 0.90 0.88]);
    axis(ax, 'equal');
    axis(ax, 'off');
    xlim(ax, [-0.1, 3.1]);
    ylim(ax, [-0.1, 4.1]);
    title(ax, 'Best Greedy Path', 'Color', TEXT, 'FontSize', 16, 'FontWeight', 'bold');

    % ── Legend ──
    h1 = patch(ax, NaN, NaN, SCROLL_C, 'EdgeColor', 'none', 'DisplayName', 'R2 Real Scroll ');
    h2 = patch(ax, NaN, NaN, FAKE_C,   'EdgeColor', 'none', 'DisplayName', 'Fake Scroll ');
    h3 = patch(ax, NaN, NaN, R1_C,     'EdgeColor', 'none', 'DisplayName', 'R1 Scroll ');
    h4 = patch(ax, NaN, NaN, EMPTY_C,  'EdgeColor', 'none', 'DisplayName', 'Empty ');
    legend(ax, [h1, h2, h3, h4], ...
        'Location',    'southoutside', ...
        'Orientation', 'horizontal', ...
        'TextColor',   TEXT, ...
        'Color',       FIGBG, ...
        'EdgeColor',   'none', ...
        'FontSize',    11);
    hold(ax, 'on');

    % ── Draw all 12 grid cells ──
    % ── Draw all 12 grid cells ──
    for b = 1:NUM_BLOCKS
        [row, col] = block_to_rc(b, COLS);
        y = ROWS - 1 - row;

        if ismember(b, env_cfg.r2_real)
            fc = SCROLL_C;
        elseif b == env_cfg.fake
            fc = FAKE_C;
        elseif ismember(b, env_cfg.r1)
            fc = R1_C;
        else
            fc = EMPTY_C;
        end

        patch(ax, [col+0.05, col+0.95, col+0.95, col+0.05], ...
                  [y+0.05,   y+0.05,   y+0.95,   y+0.95], fc, ...
              'EdgeColor', [0.33 0.33 0.33], 'LineWidth', 1.5, ...
              'FaceAlpha', 1.0, 'HandleVisibility', 'off');

        text(ax, col+0.5, y+0.5, num2str(b), ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment',   'middle', ...
             'FontSize', 18, 'FontWeight', 'bold', 'Color', 'white');
    end

    % ── Draw the best path line + START label + arrowhead ──
    if ~isempty(best_run)
        best_path_blocks = [];
        for i = 1:numel(best_run.path)
            s = best_run.path{i};
            if isnumeric(s)
                best_path_blocks(end+1) = s;
            end
        end

        if numel(best_path_blocks) >= 2
            px = zeros(1, numel(best_path_blocks));
            py = zeros(1, numel(best_path_blocks));
            for i = 1:numel(best_path_blocks)
                [ri, ci] = block_to_rc(best_path_blocks(i), COLS);
                px(i) = ci + 0.5;
                py(i) = (ROWS - 1 - ri) + 0.5;
            end

            
            % Path line
            plot(ax, px, py, '-', 'Color', ACCENT, 'LineWidth', 2.5, ...
                 'HandleVisibility', 'off');

            % START dot and label
            plot(ax, px(1), py(1), 'o', ...
                 'MarkerSize', 14, 'MarkerFaceColor', [0.2 1.0 0.4], ...
                 'MarkerEdgeColor', 'white', 'LineWidth', 1.5, ...
                 'HandleVisibility', 'off');
            text(ax, px(1)+0.05, py(1)+0.18, 'START', ...
                 'Color', [0.2 1.0 0.4], 'FontSize', 9, 'FontWeight', 'bold');

            % Arrowhead annotation at path end
            ax_pos = get(ax, 'Position');
            xl = xlim(ax);  yl = ylim(ax);

            x1 = px(end-1);  y1 = py(end-1);
            x2 = px(end);    y2 = py(end);
            xmid = (x1 + x2) / 2;
            ymid = (y1 + y2) / 2;

            fx1 = ax_pos(1) + (xmid - xl(1)) / (xl(2)-xl(1)) * ax_pos(3);
            fy1 = ax_pos(2) + (ymid - yl(1)) / (yl(2)-yl(1)) * ax_pos(4);
            fx2 = ax_pos(1) + (x2   - xl(1)) / (xl(2)-xl(1)) * ax_pos(3);
            fy2 = ax_pos(2) + (y2   - yl(1)) / (yl(2)-yl(1)) * ax_pos(4);

            annotation(fig, 'arrow', [fx1 fx2], [fy1 fy2], ...
                'Color',      ACCENT, ...
                'LineWidth',  2.5, ...
                'HeadWidth',  16, ...
                'HeadLength', 16);
        end
    end

    % Save figure
    saveas(fig, 'r2_best_path.png');
    fprintf('  Plot saved -> r2_best_path.png\n');
end
