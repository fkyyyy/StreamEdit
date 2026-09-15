# ICLR Methodology Outline: Interaction-Role-Conditioned Streaming Video Editing

> 写作定位：面向严格审稿的 Method 大纲，而非录用承诺。“达到 6 分”在这里指争取形成贡献清楚、实现一致、实验证据可检验的投稿；评分依赖相关工作比较和实际结果，不能靠叙述保证。
>
> 证据状态：作者反馈接线后的结果视觉效果不错，并已完成编辑数据用于机械臂训练和真机测试。本文档尚未据此核验视频、日志、定量表格或机器人实验协议，因此不写提升数值，也不把初步观察升级为已证明的机制。
>
> 版本边界：本大纲描述在 `7463ea6` 基础上、经过 native-attention 接线修复的完整设计，即 `s1m2_attention_mode=full`。撰写时接线代码、四组脚本和测试仍是本地未提交改动；本次文档提交不包含它们。仅检出本次文档提交不能复现修复版完整系统。正式投稿必须补充实际实验代码提交号、配置和日志，不得将旧默认配置的结果直接归因于 M2。

## 0. 严格审稿人的结论与贡献边界

**建议保留完整方法，但收敛成一个问题、两项算法贡献、一项下游验证。**

- 问题：手—物体交互视频中，目标外观需要改变，但手部、非目标区域及操作过程需要尽量保留；同时，跨帧外观记忆不能被不可靠的交互界面污染。
- 算法贡献一：共享的软交互角色表示及其驱动的编辑控制。角色结合语义、手部时序占用、源特征对应和模型速度差，而非仅对二值 mask 进行膨胀；不同角色对应不同源重建残差策略，并在不确定时回退。
- 算法贡献二：角色条件化的非对称外观记忆。较宽的读取权限与保守的一次性写入分离；通过源特征地址比较期望与当前的目标减源 value 响应，只补偿两者差异。
- 下游贡献：用编辑数据进行机器人学习并评估真机泛化。除非另有实际算法，不称其为新的机器人学习方法。

**主要审稿风险**：若同一角色图、同一先验和相同生成预算下，复杂控制并不优于普通软门控，论文会被视为启发式组合；若 M2 只在单个视频上看起来更好，也不足以支持跨帧记忆贡献。不要再通过添加无关模块或更换术语弥补这一风险。

## 1. 推荐的主文结构

| 小节 | 核心问题 | 主文必须交代 | 放入附录 |
|---|---|---|---|
| 3.1 Problem Setup and Streaming Backbone | 输入、输出和继承内容是什么？ | 手部先验、目标外观编辑、双分支与因果块 | VAE 对齐、基座配置、完整 schedule |
| 3.2 Causal Interaction-Role Estimation | 哪些位置编辑、保留或部分约束？ | 多证据物体分数、四角色、时序与连通限制 | 分位数、可靠度、面积预算、追踪细节 |
| 3.3 Role-Conditioned Editing Control | 角色如何改变实际更新？ | 对抗残差过滤、角色残差策略、熵回退、空间 Q/K 混合 | 精确索引、dtype、参数范围 |
| 3.4 Asymmetric Appearance Memory and Response-Error Correction | 如何维持外观而不反复注入错误？ | 非对称读写、冻结的源地址 ΔV bank、双检索与有界误差 | top-k、匹配阈值、复杂度、未匹配回退 |
| 3.5 Editing for Robotic Learning | 编辑数据如何进入训练？ | 数据构造、标签复用假设、训练接口 | 任务、policy、划分与评估协议 |

叙述顺序是“问题 → 共享表示 → 当前更新 → 跨帧记忆 → 下游使用”，不是按照仓库文件和历史版本逐个介绍模块。

### 可供扩写的英文总览

> We study appearance editing in videos with hand–object interactions, where changing the manipulated object must be balanced against preserving the surrounding interaction. Built on a causal streaming video editor, our method constructs a shared soft role representation from hand occupancy, semantic attention, source-feature correspondence, and source–target denoising disagreement. The roles determine reconstruction-residual control and spatial query–key blending, and define asymmetric access to an immutable appearance-residual memory. Memory reads compare a canonical target-minus-source response with the current response in source-feature coordinates and apply a bounded correction to their difference. We evaluate the resulting edited data for downstream robotic learning, with supervision validity assessed separately from visual editing quality.

避免在 Method 总览中提前写“显著提升”“保证物理一致性”“首次”，这些分别需要结果、理论或相关工作支持。

## 2. 3.1 Problem Setup and Streaming Backbone

### 2.1 输入、输出与任务范围

输入为源视频 $X^s$、源与目标文本 $c_s,c_t$、对应物体词，以及外部提供的手部 mask $M^h$。输出为目标外观编辑视频 $X^t$。

目标是在改变被操作物体外观时，尽量保留手部、背景和操作过程。当前示例是金属锅铲到木质锅铲。不要把这一设置推广成任意几何、功能或动力学变化都允许复用原动作标签。

明确先验条件：使用手部 mask；默认不使用外部物体 mask，不使用 RGB optical flow。手部 mask 的获取方式、成本和是否用到未来帧须在实验设置补充。因果生成不自动意味着整条数据预处理链严格在线。

### 2.2 符号与继承的计算

- $b$：视频时间块；$t$：latent 视频帧；$i$：空间 token；$\ell$：Transformer 层。
- $\tau$：去噪噪声时间，区别于视频时间；$k$：去噪步骤索引。
- $x^s,x^t_\tau$：源 clean latent 与目标 noisy latent。
- $v^s_\tau,v^t_\tau$：源、目标分支的 flow/diffusion 速度预测。
- $p_i$：物体归属软分数；$h_i$：手部 occupancy；$q_i^r$：角色权重；$u_i$：角色熵。

继承基座的因果 VAE、源/目标双分支、流式历史 KV、噪声构造与重新加噪采样、原生 Q/K 混合以及后期源背景 KV 注入。这些不是新增贡献。首次介绍时引用实际采用的 StreamEdit/Self-Forcing/Wan 基座，引用信息由作者核对。

“Velocity”是噪声时间中的模型更新方向，不是二维 optical flow 或物理速度。“Causal”在本方法中主要指时间块历史依赖，不代表已识别因果机制。

## 3. 3.2 Causal Interaction-Role Estimation

### 3.1 对齐的手部证据，而非简单 mask 膨胀

按照 causal VAE 的像素帧分组，分别构建：

1. **Union**：组内曾被手覆盖的位置，用于邻近与连接候选。
2. **Occupancy**：组内平均占用率，用于软角色。
3. **Persistent support**：持续手部区域，用于保守排除。

首个 causal latent 对应首帧，后续按压缩步幅分组。不得把 union 当成 occupancy；移动手部的大范围 union 不等于整片区域一直是手。默认持续占用门限为 1.0，空间投影和排除规则放入附录。

### 3.2 多证据物体分数

以 clean-source 的物体词 attention 提供语义响应，结合手部邻近、可见性判断和局部传播形成候选。邻近半径由手部面积自适应估计，而非所有场景固定同一像素尺度。

以手附近和远处语义响应的分离度估计 $R_{\mathrm{att}}$，据此调节种子、候选门限和控制强度。时间传播利用选定层 clean-source query 特征的 top-4 相似匹配：

$$
\widetilde p_{t,i}=c_{t,i}\sum_{j\in\mathcal N_4(i)}w_{ij}p_{t-1,j}.
$$

其中匹配权重、温度和置信项由特征相似性确定；传播与当前观测结合，而不是以传播无条件覆盖观测。短暂观测缺失时仅在匹配证据足够时恢复支持，恢复状态不取得记忆写入权限。

**术语约束**：称为 evidence reliability / online adaptation，不声称分数具有经过验证的概率校准意义；特征匹配也不是光流。

### 3.3 连通与面积限制

将候选约束为手部锚定的连通支持，并结合前一支持的空间重叠抑制跳到邻近干扰物。实际面积预算结合种子面积、手部占用、proposal、可靠度及历史参考；覆盖率 0.18 是上限，不是固定物体面积。

该模块的论证重点是防止控制权限扩散，而不是宣称得到精确物体分割。前一支持的空间重叠不是特征输运，应与前述 query 匹配区分。

### 3.4 以模型编辑响应细化，而非以速度直接四分类

每块首次双分支去噪预测后计算：

$$
F=\operatorname{Pool}\left[\operatorname{QNorm}\left(\operatorname{Mean}_c|v^t_\tau-v^s_\tau|\right)\right].
$$

比较种子区域与周围环带的中位数响应，并以 MAD 归一化，得到场可靠度。利用似然式权重削弱不支持物体假设的位置，并在语义、邻近、手部占用和可见性约束内补充候选。最终仍受先前选定的 connected support 限制，不能写成速度场能自由发现全图新物体。

源/目标分支可能具有不同 latent 和历史，因此 $v^t-v^s$ 是分支编辑响应差，**不是严格的 same-latent controlled intervention**。本模块不另增一次专门的反事实去噪查询；但系统仍有 clean-source probing 等成本，不可声称零开销。

### 3.5 共享的四角色表示

在对齐网格上定义：

$$
q_i^o=p_i(1-h_i),\qquad q_i^c=p_i h_i,\qquad
q_i^h=(1-p_i)h_i,\qquad q_i^b=(1-p_i)(1-h_i).
$$

分别对应物体核心、交互界面、手部和背景，且 $\sum_r q_i^r=1$。交互界面在代码中名为 `boundary`，是 latent 级手部与物体证据重叠的代理，不是真实物理接触标签。

$$
u_i=-\frac{1}{\log 4}\sum_r q_i^r\log(q_i^r+\epsilon).
$$

角色熵用于保守回退与读取抑制，不应被解释成经过标注集校准的错误概率。角色投影到不同控制网格时联合插值并重新归一化。

**实现细节边界**：代码还维护独立的 clean-source causal owner tracker，并将其支持并入部分前景/KV 元数据。该 tracker、角色后验的时间传播、用于 M2 的连通支持不是同一个变量。正文以共享角色为主线，附录明确这些不同支持的接口，不能宣称所有 KV 元数据均被同一角色图无例外替换。

## 4. 3.3 Role-Conditioned Editing Control

### 4.1 为什么不能统一保留源残差

在基座的线性噪声插值参数化下，源重建速度为 $v^{s,*}=\epsilon-x^s$。定义：

$$
r=v^{s,*}-v^s_\tau,\qquad d=v^t_\tau-v^s_\tau.
$$

$r$ 提供源重建约束，但可能与目标外观更新冲突；直接把全部源信息屏蔽，又可能损伤交互附近的保留约束。

### 4.2 过滤对抗分量

沿 latent channel 方向计算：

$$
r_{\mathrm{safe}}=r-\frac{\min(\langle r,d\rangle,0)}{\|d\|_2^2+\epsilon}d.
$$

只在编辑方向有效时应用；方向近零时保留原残差。该操作抑制对抗 $d$ 的残差投影，保留其他分量。不要声称它严格分离“几何”和“外观”，也不要在忽略数值稳定项的情况下宣称精确正交或理论保证。

### 4.3 角色控制与不确定性回退

$$
v_{\mathrm{role}}=v^t_\tau+
(\alpha_o q^o+\alpha_c q^c)r_{\mathrm{safe}}+(q^h+q^b)r.
$$

默认 $\alpha_o=0.10$、$\alpha_c=0.35$：物体核心较少保留源残差，交互界面保留更强约束，手部与背景使用完整源残差。这里依旧以 $v^t_\tau$ 为基底，因此不等价于像素级精确复制源手部或背景。

保留基座的动态背景策略：

$$
B=1-\operatorname{MinMax}(\operatorname{Mean}_c|v^t_\tau-v^s_\tau|),\qquad
v_{\mathrm{native}}=v^t_\tau+Br,
$$

$$
v_{\mathrm{final}}=(1-u)v_{\mathrm{role}}+u v_{\mathrm{native}}.
$$

源重建残差在角色明确时按角色使用，不确定时回到基座，而不是强迫硬分类。当前 clean latent 预测为 $\widehat x^t=x^t_\tau-\tau v_{\mathrm{final}}$，后续沿用基座重新加噪步骤，不要擅自写成不同的 ODE 积分器。

### 4.4 空间 Q/K 混合：同一权限的注意力接口

利用下一小节定义的角色读取权重 $g_i^{\mathrm{read}}$，将基座标量混合率扩展为空间混合率：

$$
\beta_i=\beta+(1-\beta)\eta g_i^{\mathrm{read}},\qquad
Q_i=\beta_iQ_i^t+(1-\beta_i)Q_i^s,\qquad
K_i=\beta_iK_i^t+(1-\beta_i)K_i^s.
$$

这是角色表示的另一个控制接口，不必再包装成第三项独立核心创新。当前 Q/K 使用空间率，历史 key 仍按基座标量率；目标 value 不因该操作被直接替换。

实现使用 $\beta=1-\tau_{\mathrm{next}}^{\rho}$、$\rho=2$、$\eta=1$ 构造并存储空间率。它在当前 forward 后设置，供后续 forward 使用；每块首次 forward 尚无该覆盖量。主文给出上面的混合形式，附录按照代码写明延后一拍的时间索引，不把角色细化逆向应用到已完成的 forward。

## 5. 3.4 Asymmetric Appearance Memory and Response-Error Correction

### 5.1 读写的不对称代价

读取需要覆盖物体及部分交互界面，写入则需避免把不可靠外观永久固定。令 $S_i$ 为连通支持：

$$
g_i^{\mathrm{read}}=S_i(q_i^o+\lambda_cq_i^c)(1-u_i),\qquad \lambda_c=0.5.
$$

小于 0.20 的读取权重置零。写入候选满足：

$$
q_i^o\ge0.50,\quad q_i^o\ge q_i^c,\quad q_i^h\le0.10,\quad u_i\le0.50,\quad S_i=1.
$$

按 $q_i^o(1-u_i)(1-q_i^c)(1-q_i^h)$ 排序，根据证据可靠度保留合格候选中约 10%–30% 的高分 token，有候选时至少保留一个。附录给出 $R_{\mathrm{write}}=\sqrt{R_{\mathrm{att}}(0.25+0.75R_{\mathrm{temp}})}$ 和实际取整规则。

恢复帧禁止写入，并在本次调用的 latent-frame block 内施加：

$$
W_t=\widehat W_t\cap\operatorname{Dilate}_1(\widehat W_{t-1}),\qquad W_0=0.
$$

这不是跨所有 blocks 维护的持久两帧确认，也不是 feature warping。允许满足资格条件的混合 token 写入，不能笼统写成“所有 contact probability 非零的位置都禁止写入”。

### 5.2 冻结的源地址外观残差

每块去噪完成后进行 clean target commit。第一次出现非空合格支持时，对选定层存储：

$$
\mathcal M^\ell=\{(K_{s,j}^{\ell,\mathrm{clean}},\Delta V_j^\ell):j\in W\},\qquad
\Delta V_j^\ell=V_{t,j}^{\ell,\mathrm{clean}}-V_{s,j}^{\ell,\mathrm{clean}}.
$$

源 key 是 pre-RoPE 地址；payload 是目标减源 value，而非完整目标 value。bank 在建立后冻结，不从后续生成结果递归更新。

写作上称为 **immutable source-addressed appearance-residual memory**，不要称为在线学习的长期记忆。如果第一块无合格核心，冻结延期；如果始终无合格支持，则整个视频没有 M2 读取。该取舍减少递归写入带来的污染机会，但也无法主动修复初始错误 anchor 或覆盖所有未见视角。

### 5.3 在共同源地址中比较两种响应

用当前 clean-source query 分别检索：

$$
D_{\mathrm{desired}}=\mathcal R(Q_s^{\mathrm{clean}},K_s^{\mathrm{canonical}},\Delta V^{\mathrm{canonical}}),
$$

$$
D_{\mathrm{current}}=\mathcal R(Q_s^{\mathrm{clean}},K_s^{\mathrm{current,clean}},V_t^\tau-V_s^\tau).
$$

$\mathcal R$ 先以跨 head 展平的 cosine similarity 选取 top-8 且相似度至少 0.35 的地址，再以逐 head scaled dot-product 权重聚合 payload。当前响应只从 $g^{\mathrm{read}}>0$ 的 key 支持读取。

**重要区别**：地址 Q/K 来自 clean source；canonical payload 来自 clean source/target；current payload 来自当前 noisy denoising forward。不要将两个响应都写成 clean feature 差。

### 5.4 有界的响应差校正

$$
e=D_{\mathrm{desired}}-D_{\mathrm{current}}.
$$

逐 query 以 head/channel RMS 对误差限幅：

$$
\bar e=e\min\left(1,\frac{\gamma\max(\operatorname{RMS}(D_{\mathrm{desired}}),\operatorname{RMS}(D_{\mathrm{current}}))}{\operatorname{RMS}(e)+\epsilon}\right),\qquad\gamma=1.
$$

其中公式中的分母稳定写法为简化表达，复现时依代码的 `clamp_min`。令 $c$ 为两次匹配中较低的最佳 cosine similarity 经阈值归一化所得的置信权重，$a$ 表示两次检索均匹配：

$$
O'=O_{\mathrm{native}}+\lambda\,a\,g^{\mathrm{read}}c\,\bar e,\qquad\lambda=0.20.
$$

强调区别：不是每步盲目叠加同一份目标 ΔV，而是反馈当前尚未实现的响应差。无 bank、未匹配、无权限或误差为零时不校正；已有数值测试覆盖相关退化情形。

“Closed-loop”仅指响应差反馈，不证明最终视频误差单调下降。$e$ 是检索到的 value 响应差，加入的是 attention output，不能把代数诊断量 $e-\lambda g c\bar e$ 直接称为重新执行模型后测得的真实误差收敛。

M2 校正算子本身不修改原生 attention softmax 分母及 KV payload，但完整系统仍有空间 Q/K 混合、原生源背景注入和 clean target history 更新，不能笼统说“整个 attention/KV 不变”。

## 6. 3.5 Editing for Robotic Learning

此节保持短小，具体协议放 Experiments。作者确认已做真机训练和测试，但以下内容必须补齐后才能写为正式方法。

若原始示教为 $\mathcal D=\{(o_{1:T}^{(n)},a_{1:T}^{(n)})\}$，只有确认编辑不改变动作语义及时间对齐后，才能写：

$$
\widetilde{\mathcal D}=\{(\mathcal E(o_{1:T}^{(n)},c^{(m)}),a_{1:T}^{(n)})\}_{n,m}.
$$

- `[待补充]` 视频是人类演示还是机器人示教；动作与状态标签的来源。
- `[待补充]` 时间、相机、观测模态的对齐方式和编辑失败样本处理。
- `[待补充]` 外观变化范围，是否涉及几何、接触、材料动力学变化，以及标签复用依据。
- `[待补充]` policy、训练目标、真实/编辑数据比例、训练预算与随机种子。
- `[待补充]` 未见外观/物体/背景等具体泛化划分、真机次数、成功定义和不确定性。

不能在动作来源未知时自行补写 imitation-learning loss，也不能把训练/测试同一原始轨迹的不同编辑版本当作独立泛化样本。划分须按原始轨迹及实际泛化目标避免泄漏。

## 7. 算法框图与伪代码骨架

建议图示只画三个主体：**role estimation → editing control / asymmetric memory**，并区分视频时间和噪声时间。记忆写入箭头仅在 clean commit 后出现，冻结后不再循环写回。

```text
Inputs: source video, source/target prompts, hand evidence, frozen editing backbone
Initialize: temporal role state, causal owner state, native KV histories, empty ΔV bank
For each causal video block:
    Encode/project aligned hand evidence
    Run clean-source probe; capture semantic attention and source Q/K/V
    Infer role prior with temporal correspondence, connected support and area control
    Update causal-owner metadata; derive initial read/write permissions
    For each denoising step:
        Run source and target branches with native history
        Consume stored spatial Q/K rates if available and enabled
        If M2 enabled and bank frozen: apply matched, gated response-error correction
        On the first step: refine roles using source–target velocity disagreement
        Update role-derived permissions
        Filter antagonistic source residual; apply role policy and entropy fallback
        Predict clean target and follow the backbone re-noising schedule
        Store spatial blending rates for the subsequent forward
    Commit clean target KV
    If bank empty and write support nonempty: freeze source keys and clean target–source ΔV
Decode edited video
```

角色推断先验、首次速度细化和 M2 的先后顺序必须保留：当前 forward 的角色细化不能反过来影响已完成的 attention。不要将所有步骤画成无时间依赖的同时计算。

## 8. 支撑“有竞争力的投稿”所需证据

### 8.1 先证明实现与实验相符

- 使用修复版，报告明确的实验提交号、命令、输入和模型版本。
- 核查 `S1M2_ATTENTION` 的实际 native/spatial/M2 调用次数。
- 分别报告 bank 延迟冻结、始终未冻结、读取未匹配、匹配但零误差、实际非零校正的情况。
- 不把 CPU 单元测试、命令行开启参数或成功写入 bank 当成视频有效性证据。

### 8.2 当前四组实验回答什么

| 组别 | 固定基础 | M2 | 空间 Q/K | 回答的问题 |
|---|---|---|---|---|
| A: legacy | 角色、S1、写入策略、基座 | 关 | 关 | 旧实际路径表现 |
| B: m2 | 同 A | 开 | 关 | M2 的增量效果 |
| C: spatial | 同 A | 关 | 开 | 空间混合的增量效果 |
| D: full | 同 A | 开 | 开 | 完整系统与交互效应 |

固定输入、prompt、seed、采样步数与 backbone；报告新增模块实际耗时和显存，不能因步数相同就称总计算量完全相同。角色和 bank 的构建规则固定，不等于最终张量相同：空间混合和反馈可能改变后续速度、角色及写入内容。A/B/C/D 是系统消融；如需排除定位变化带来的解释，再做固定/回放角色支持的机制对照，并明确该对照与实际在线系统的区别。

A/B/C/D 本身不能证明全部角色模块的新意。还需基座、同角色图普通软门控、去掉方向过滤、无熵回退、对称读写、无恢复禁写/时间一致性，以及 open-loop ΔV addition 等有针对性的对照；优先选择能够直接排除替代解释的实验，而非盲目穷举全部组合。

### 8.3 机制指标不能只测视觉美观

- **编辑成功**：目标对象的语义/外观符合度，不能只用整帧相似度。
- **保留质量**：手部和非目标区域相对源视频的偏移、伪影及跨帧退化。
- **时间一致性**：在源特征对应或独立轨迹对齐后衡量物体身份稳定性，并覆盖遮挡后再显露；静态相邻帧相似度容易奖励不运动。
- **记忆质量**：写入错误区域比例或人工抽查、匹配失败率、实际校正覆盖率与强度、随时间的外观回退。
- **动作有效性**：独立的交互/运动一致性评估，与最终真机成功率分开报告。
- **下游泛化**：同真实数据量、编辑数据量和 policy 训练预算，对比基座编辑与完整方法；给出多次试验及不确定性。

评估区域尽量使用独立标注或评估器，避免用方法自己的角色分数同时定义“正确区域”和打分。光流均值下降可能仅意味着运动被抹掉，亮度稳定也不是身份一致性；现有比较脚本不足以单独支撑上述机制主张。

### 8.4 相关工作的排他性论证

按三类最近邻对比，而不是宣称每个算子本身首次提出：

1. 前景门控/交互保持编辑：是否已有相同角色划分及同一表示下的更新权限？
2. attention/KV 注入和身份记忆：是否区分源地址与目标残差，是否已有同类差值反馈？
3. 机器人视觉数据增强：是否已用生成编辑改善泛化，我们新增的是何种可检验的交互保持能力？

至少回答“相同手部先验与相同区域下，我们比普通门控/直接残差注入多解决了什么”。相关工作检索完成前，不写首创断言。

## 9. 附录与复现清单

主文保留角色公式、S1 公式、读写不对称、ΔV 双检索和误差校正。附录提供：

- hand union/occupancy/persistent 的因果时间投影与空间缩放；
- attention 和 field 的分位数归一化、可靠度、可见性、posterior threshold 与 EMA；
- temporal query top-k、温度、恢复条件、连通支持与面积参考状态；
- 独立 causal-owner tracker 与各类 mask/KV metadata 的接口；
- 完整采样与空间混合索引、M2 clean/noisy 张量来源、层和 dtype；
- 写入资格、可靠度保留比例、block 内一致性及冻结时点；
- 运行时间、显存、bank 大小、处理视频长度和失败回退统计。

**复杂度须如实报告**：top-k 检索仍先计算与候选 key 的相似度；不能因最终只保留 8 个 key 就称检索为常数复杂度。对于 $A$ 个活跃 query、$M$ 个 bank key 和 $N$ 个当前 key，地址相似度主要成本随 $A(M+N)d$ 增长；一次性 bank 不随视频长度持续增长，但原生 KV 与源 probing 成本另计。CPU 连通区域处理及设备间同步成本也应包含在实际耗时中。

### 实现导航

下表路径相对本文件所在的 `Self-Forcing_StreamEdit/`，行号会随接线提交变化，以符号为准。

| 内容 | 文件 / 符号 |
|---|---|
| 手部证据投影 | `pipeline/mask_alignment.py` / `project_hand_evidence_to_causal_latents` |
| 角色先验与速度细化 | `pipeline/hand_role_inference.py` / `HandRoleInferencer` |
| 在线可靠度与面积预算 | `pipeline/adaptive_role_calibrator.py` / `AdaptiveRoleCalibrator`, `AdaptiveOwnerExtentController` |
| 角色残差与读写权限 | `pipeline/role_router.py` / `PosteriorResidualFlowRouter`, `build_role_memory_gates` |
| 残差方向过滤 | `pipeline/appearance_leakage.py` / `remove_antagonistic_source_residual` |
| 独立 owner 状态 | `pipeline/causal_ownership.py` / `CausalObjectOwnershipTracker` |
| 一次冻结的 bank | `pipeline/immutable_delta_v_bank.py` / `ImmutableDeltaVBank` |
| M2 数值算子 | `wan/modules/attention.py` / `closed_loop_delta_v_memory_attention` |
| 修复后的实际 attention 接线 | `wan/modules/causal_model.py` / `CausalWanSelfAttention.forward` |
| 时序编排、S1、commit 和调用诊断 | `pipeline/edit_causal_inference.py` / `rollout_inference`, `inference` |
| 实验入口 | `run_cook_S1M2_ablation.sh`, `run_cook_S1M2_A_legacy.sh` 至 `run_cook_S1M2_D_full.sh` |
| 接线与脚本测试 | `tests/test_s1m2_attention_modes.py` |

当前默认复现参数：15 步、seed 0、query/M2 层 8/12/16/20、$\alpha_o=0.10$、$\alpha_c=0.35$、读取 contact 权重 0.50、读取下限 0.20、物体写入资格 0.50、M2 strength 0.20、匹配下限 0.35、top-k 8、误差比上限 1.0、覆盖率上限 0.18。它们是默认值，不代表已完成跨场景共享参数验证；最终 schedule 以 scheduler 实际变换后为准。

## 10. 最终写作禁区与完成条件

不写：无 mask、精确接触识别、纯速度分类、严格因果干预、统计概率校准、无超参数、保证动作标签有效、几何/外观精确解耦、整个 KV 不变、每帧更新 immutable bank、固定第一块冻结、闭环收敛保证、已实现 2/4 步自适应调度。

保留的诚实局限：手部先验误差、小物体和多物体歧义、源特征错配、遮挡/快速运动、新视角缺少 canonical 支持、初始外观错误冻结、面积上限和阈值的适用范围，以及机器人标签复用的任务条件。

**定稿门槛**：方法段落中的每个收益必须能对应独立的实验或被降格为设计动机；每个公式必须能对应实际执行路径；完整贡献必须由完整版本的实验支撑。若证据成立，这个大纲足以组织一篇有竞争力的投稿；若证据不成立，应缩减贡献，而不是继续加强形容词。
