---
layout: post
title: Expanding the Capabilities of Reinforcement Learning via Text Feedback
date: '2026-05-14'
rating: 6
rating_label: Excellent, can't unsee it
tags:
- Machine Learning
- Reasoning Models
tag_data:
- name: Machine Learning
  color: moss
- name: Reasoning Models
  color: rose
paper_url: https://arxiv.org/abs/2602.02482
---

## Motivation 

RLVR has been a driving force in recent improvements in language models for a variety of tasks, but the each trajectory carries very little information, only providing feedback about whether the answer was correct or incorrect. If the model is unable to solve the task, it will never receive a reward of 1 that allows it to learn - RL does not accomplish anything if all samples achieve the same reward of 0, it needs to be able to see both good and bad trajectories.

Ideally, we would have correct traces for the model to match, which enters the regime of SFT and distillation. However, this data cannot be generated for frontier models and it is expensive to collect from humans. 

Text feedback collected in human-model interactions fits a unique set of criteria by already being collected at scale and providing richer feedback than binary rewards. Being able to take advantage of this data to improve model performance would unlock a new set of possibilities for model training. However, no there are no existing techniques for taking advantage of this data to improve performance *before feedback*.

This is not as simple as appending the feedback to the prompt for the model to begin revising its answer and providing rewards for each response (i.e., the multi-turn objective: <span class="math-render" data-display="inline" data-math="J_\text{MultiTurn}(\pi)=\mathbb{E}^\pi \left[ \sum_{h=0}^{H-1} r_h \right]"></span>); training human and model discussions where the 2nd, 3rd or 5th piece of feedback led to a solution would result in the model improving its ability to utilizing current feedback at test time, rather than having the model internalize the feedback during training and use the knowledge it gained for future first attempts. This is validated empirically.

This paper formalizes this idea by describing an RL from Text Feedback (RLTF) setting that focuses on the first attempt reward and provides two methods to learn in this setting (RLTF-SD and RLTF-FM).

## What they did

### Formulation

They first formalize the RLTF setting by phrasing the prompt <span class="math-render" data-display="inline" data-math="x_h"></span> at step <span class="math-render" data-display="inline" data-math="h"></span> (not individual tokens, but steps counting each (human prompt, model response) pair in the conversation) as a function of a triple:
- <span class="math-render" data-display="inline" data-math="x_{h-1}"></span>: the prompt at the previous step
- <span class="math-render" data-display="inline" data-math="y_{h-1}"></span>: the LLM output of the previous step, sampled from the policy <span class="math-render" data-display="inline" data-math="\pi(\cdot|x_{h-1})"></span>
- <span class="math-render" data-display="inline" data-math="c_{h-1}"></span>: the feedback for the previous step, sampled from the feedback distribution <span class="math-render" data-display="inline" data-math="M(\cdot|x_{h-1}, y_{h-1})"></span>

<span class="math-render" data-display="block" data-math="x_h = f(x_{h-1}, y_{h-1}, c_{h-1})"></span>

Note: If <span class="math-render" data-display="inline" data-math="y_h"></span> is a high quality response (say, with reward 1) the conversation terminates.

To isolate the role of text feedback, the objective is phrased as such:

- <span class="math-render" data-display="inline" data-math="\mu_0"></span>: the distribution over initial prompts
- <span class="math-render" data-display="inline" data-math="\pi (\cdot | x_0)"></span>: the output distribution of a model given an initial prompt
- <span class="math-render" data-display="inline" data-math="R(x_0,y)"></span>: the reward given an initial prompt and LLM output

<span class="math-render" data-display="block" data-math="J_{\text{SingleTurn}}(\pi) = \mathbb{E}_{x_0 \sim \mu} \left[ \mathbb{E}_{y \sim \pi (\cdot | x_0)} \left[ R(x_0,y) \right] \right]"></span>

This evaluates the LLM on initial prompts without additional feedback at test time. The question then becomes how to design learning objectives/algorithms that make use of feedback-augmented traces <span class="math-render" data-display="inline" data-math="x_h"></span> to increase <span class="math-render" data-display="inline" data-math="J_{\text{SingleTurn}}(\pi)"></span>.

### Self Distillation

If we view the policy acting under feedback as the correct policy which we want to model our first turn policy after, than it naturally follows that this feedback conditioned policy should take the role of a teacher.

This means we have off-policy demonstration data <span class="math-render" data-display="inline" data-math="(x_0, y_1)"></span> where <span class="math-render" data-display="inline" data-math="y_1 \sim \pi(\cdot | x_1)"></span>, which is a better quality sample than what would be sampled from <span class="math-render" data-display="inline" data-math="\pi(\cdot|x_0)"></span> because of the guidance from feedback. We want to use this to improve <span class="math-render" data-display="inline" data-math="\pi(\cdot|x_0)"></span>.

This allows us to take a common distillation perspective and apply AWR (weighting each demonstration by its relative performance (advantage) and maximize log-likelihood under the target distribution):

<span class="math-render" data-display="block" data-math="\mathbb{E}_{y_1 \sim \pi(\cdot|x_1)}\left[A(x_0, y_1)\cdot\log\pi(y_1|x_0)\right]"></span>

This is used as an auxiliary loss to <span class="math-render" data-display="inline" data-math="J_\text{MultiTurn}(\pi)"></span>.

This paper focuses on the case of termination at h=1. I am assuming this is due it being easier to procure a dataset for this case or theoretic simplicity (if considering a sample-dependent horizon H, taking a mean with a division by H introduces bias that is hard to reconcile mathematically), though I attempt to show this more general case below with a sum instead of a mean (which is still biased, but is more mathematically amenable).

We can see the AWR expression is a biased estimate of <span class="math-render" data-display="inline" data-math="J_{\text{SingleTurn}}(\pi)"></span>, which empirically performs better than the unbiased high-variance estimate:

<span class="math-render" data-display="block" data-math="J_{\text{SingleTurn}}(\pi) = \mathbb{E}_{x_0 \sim \mu, y_1 \sim \pi (\cdot | x_0)} \left[ R(x_0,y) \right]"></span>

First introduce the advantage, which is assumed to be an unbiased estimator of the reward:

<span class="math-render" data-display="block" data-math="= \mathbb{E}_{x_0 \sim \mu, y \sim \pi (\cdot | x_0)} \left[ A(x_0,y_1) \right]"></span>

Next get the policy gradient (log-derivative trick):

<span class="math-render" data-display="block" data-math="\nabla J_\text{SingleTurn}(\pi) = \mathbb{E}_{y_1 \sim \pi(\cdot|x_0)}\left[ A(x_0, y_1) \nabla\log\pi(y_1|x_0)\right]"></span>

Add importance sampling weights to make it an expectation conditioned on the prompt with feedback:

<span class="math-render" data-display="block" data-math="=\mathbb{E}_{y_1 \sim \pi(\cdot|x_1)}\left[\frac{\pi(y_1|x_0)}{\pi_\text{ref}(y_1|x_1)} A(x_0, y_1) \nabla\log\pi(y_1|x_0)\right]"></span>

Remove the importance weights by setting <span class="math-render" data-display="inline" data-math="\pi_\text{ref}(y_1|x_1) = \pi(y_1|x_0)"></span>:

<span class="math-render" data-display="block" data-math="\nabla\ell_\text{awr} = \mathbb{E}_{y_1 \sim \pi(\cdot|x_1)}\bigl[A(x_0, y_1)\nabla\log\pi(y_1|x_0)\bigr]"></span>

The last step introduces bias, but removes the variance introduced by the importance weights. I found it interesting that low variance performs better than unbiasedness in an LLM setting; I had always assumed that we would prefer unbiasedness because of how the variance of means scales with <span class="math-render" data-display="inline" data-math="\frac{1}{n}"></span> due to LLN, which seems great in settings with large amounts of data, and the fact that losses are generally defined as expectations that we approximate with a mean in practice. However, I do remember Nathan Lambert at Ai2 also talking about finding gains when removing importance sampling when training OLMo models.

The paper does not show the case of arbitrary h with trajectories that have a sample-dependent horizon H, so I will attempt to work it out.

Let us denote <span class="math-render" data-display="inline" data-math="\tau = (y_0, c_0, y_1, c_1, \ldots, y_H)"></span> as the full trajectory without the input prompt (and is deterministic given <span class="math-render" data-display="inline" data-math="x_0"></span> because each <span class="math-render" data-display="inline" data-math="x_h"></span> is deterministic function of <span class="math-render" data-display="inline" data-math="x_{h-1}, y_{h-1}, c_{h-1}"></span>) and the trajectory distribution as <span class="math-render" data-display="inline" data-math="\mathbb{P}^\pi(\tau|x_0) = \prod_{h=0}^{H} \pi(y_h|x_h)M(c_h|x_h, y_h)"></span>. The AWR expression for the case including all <span class="math-render" data-display="inline" data-math="h"></span> with sample-dependent horizon <span class="math-render" data-display="inline" data-math="H"></span> is:

<span class="math-render" data-display="block" data-math="\mathbb{E}_{x_0 \sim \mu, \tau \sim \mathbb{P}^\pi(\cdot|x_0)}\left[\sum_{h=1}^{H} A(x_0, y_h)\nabla\log\pi(y_h|x_0)\right]"></span>

Splitting the sum gives:

<span class="math-render" data-display="block" data-math="=\mathbb{E}_{x_0 \sim \mu} \left[ \sum_{h=1}^{\infty} \mathbb{E}_{\tau \sim \mathbb{P}^\pi(\cdot|x_0)}\left[\mathbf{1}[H \geq h] \cdot A(x_0, y_h)\nabla\log\pi(y_h|x_0)\right] \right]"></span>

Writing <span class="math-render" data-display="inline" data-math="\mathbf{1}[H \geq h] = \prod_{k=0}^{h-1}\mathbf{1}[R(x_0, y_k) &lt; 1]"></span> makes each term in the sum becomes:

<span class="math-render" data-display="block" data-math="\mathbb{E}_{\tau \sim \mathbb{P}^\pi(\cdot|x_0)}\left[ \left(\prod_{k=0}^{h-1}\mathbf{1}[R(x_0, y_k) &lt; 1] \right)\cdot A(x_0, y_h)\nabla\log\pi(y_h|x_0)\right]"></span>

Out of all variables in <span class="math-render" data-display="inline" data-math="\tau \sim \mathbb{P}^\pi(\cdot|x_0)"></span>, <span class="math-render" data-display="inline" data-math="A(x_0, y_h)\nabla\log\pi(y_h|x_0)"></span> depends only on <span class="math-render" data-display="inline" data-math="y_h"></span>, and the indicators only depend on <span class="math-render" data-display="inline" data-math="y_0,...,y_{h-1}"></span>, so the natural next step is to condition on all variables in <span class="math-render" data-display="inline" data-math="\tau"></span> that precede <span class="math-render" data-display="inline" data-math="y_h"></span>, so <span class="math-render" data-display="inline" data-math="\tau"></span> excluding <span class="math-render" data-display="inline" data-math="y_h,...,y_H"></span> and <span class="math-render" data-display="inline" data-math="c_h,...,c_{h+1}"></span>, which I will denote as <span class="math-render" data-display="inline" data-math="\xi_h=(y_0, c_0, \ldots, y_{h-1}, c_{h-1})"></span>:

<span class="math-render" data-display="block" data-math="= \mathbb{E}_{\xi_h|x_0}\left[\mathbb{E}_{y_h \mid \xi_h,x_0}\left[\left(\prod_{k=0}^{h-1}\mathbf{1}[R(x_0, y_k) &lt; 1] \right)\cdot A(x_0, y_h)\nabla\log\pi(y_h|x_0)\right]\right]"></span>

The distribution of <span class="math-render" data-display="inline" data-math="y_h"></span> given <span class="math-render" data-display="inline" data-math="\xi_h"></span> is <span class="math-render" data-display="inline" data-math="\pi(\cdot|x_h)"></span> because <span class="math-render" data-display="inline" data-math="x_h"></span> is a deterministic function of <span class="math-render" data-display="inline" data-math="\xi_h"></span> and <span class="math-render" data-display="inline" data-math="y_h"></span> is defined as depending only on <span class="math-render" data-display="inline" data-math="x_h"></span>:

<span class="math-render" data-display="block" data-math="= \mathbb{E}_{\xi_h|x_0}\left[\left(\prod_{k=0}^{h-1}\mathbf{1}[R(x_0, y_k) &lt; 1] \right)\cdot \mathbb{E}_{y_h \sim \pi(\cdot|x_h,x_0)}\left[A(x_0, y_h)\nabla\log\pi(y_h|x_0)\right]\right]"></span>

From here we can use the same argument as earlier:

<span class="math-render" data-display="block" data-math="= \mathbb{E}_{\xi_h|x_0}\left[\left(\prod_{k=0}^{h-1}\mathbf{1}[R(x_0, y_k) &lt; 1] \right)\cdot \nabla J_\text{SingleTurn}(\pi, x_0)\right]"></span>

Where we have

<span class="math-render" data-display="block" data-math="J_\text{SingleTurn}(\pi, x_0) = \nabla\mathbb{E}_{y \sim \pi(\cdot|x_0)}[R(x_0, y)] = \mathbb{E}_{y_h \sim \pi(\cdot|x_h)}\left[A(x_0, y_h)\nabla\log\pi(y_h|x_0)\middle|x_0, \xi_h\right]"></span>

Writing out the full expression:

<span class="math-render" data-display="block" data-math="\mathbb{E}_{x_0 \sim \mu} \left[ \sum_{h=1}^{\infty} \mathbb{E}_{\xi_h|x_0}\left[\left(\prod_{k=0}^{h-1}\mathbf{1}[R(x_0, y_k) &lt; 1] \right)\cdot \nabla J_\text{SingleTurn}(\pi, x_0)\right] \right]"></span>

<span class="math-render" data-display="block" data-math="= \mathbb{E}_{x_0 \sim \mu}\left[\sum_{h=1}^{\infty} P_{\xi_h|x_0}(H \geq h) \cdot \nabla J_\text{SingleTurn}(\pi, x_0)\right]"></span>

<span class="math-render" data-display="block" data-math="= \mathbb{E}_{x_0 \sim \mu}\left[\mathbb{E}[H|x_0] \cdot \nabla J_\text{SingleTurn}(\pi, x_0)\right]"></span>

We cannot simplify further, because the condition expectation and gradient co-vary with <span class="math-render" data-display="inline" data-math="x_0"></span>. We would need to know the joint distribution of <span class="math-render" data-display="inline" data-math="H"></span> (the discussion length) and the single turn loss gradient. We can probably assume that harder prompts will tend to have both larger <span class="math-render" data-display="inline" data-math="\mathbb{E}[H|x_0]"></span> (the policy fails more often, so episodes run longer) and larger <span class="math-render" data-display="inline" data-math="\nabla J_\text{SingleTurn}(\pi, x_0)"></span> (there's more room to improve). The multi-turn AWR estimator implicitly upweights the gradient contributions from harder prompts, which is arguably a sensible inductive bias.

Going back to the 2nd turn setting, the paper finds that the choice of <span class="math-render" data-display="inline" data-math="A(x_0, y_1)"></span> is important. If we use the standard GRPO choice of <span class="math-render" data-display="inline" data-math="A(x_0,y_1^i) = R(x_0,y_1^i) - \frac{1}{N} \sum_{j=1}^N R(x_0,y_1^j)"></span>, we are at risk of no/low learning signal. Intuitively, RL needs to be able to distinguish good and bad samples, but the teacher can be expected to be highly reliable, so the rewards can sometimes be constant: the probability that the rewards are constant is <span class="math-render" data-display="inline" data-math="p_1^N + (1-p_1)^N"></span>, given that the probability that second turn policy gives an output with reward 1 is <span class="math-render" data-display="inline" data-math="p_1"></span>. This quantity is ~0.35 with <span class="math-render" data-display="inline" data-math="p_1=0.9,N=10"></span> and <span class="math-render" data-display="inline" data-math="p_1=0.95,N=20"></span>. With higher <span class="math-render" data-display="inline" data-math="N"></span>, the learning signal is still low if the rewards are nearly constant.

This does not match our expectations; we want a poorly performing first turn model to learn from a high performing second turn model. This motivates the idea of using the mean of the first turn rewards: <span class="math-render" data-display="inline" data-math="A(x_0,y_1^i) = R(x_0,y_1^i) - \frac{1}{N} \sum_{j=1}^N R(x_0,y_0^j)"></span>. This baseline will naturally not be equal to the second turn rewards, and it is also nice because it actually reflects the baseline we are attempting to improve. The expectation of the average term is still 0 so the advantage stays unbiased.

They also try a Rejection Sampling for distillation approach by setting <span class="math-render" data-display="inline" data-math="A(x_0, y_1) = R(x_0, y_1) \cdot \mathbf{1}[R(x_0, y_1) &gt; R(x_0, y_0)]"></span>, discarding negative samples (gradients are 0). This underperforms.

### Feedback Modeling

The feedback has a specific quality that the second-turn policy doesn't: it captures the user's intent and can therefore be treated as always correct, unlike the the second-turn model behavior which requires a reward, constraining supervision to a scalar. If we modeled the feedback provider instead, we could have dense token-level supervision (because all tokens in the feedback are correct) that specifically addresses the flaws of each output at every turn.

This is used as an auxiliary loss with <span class="math-render" data-display="inline" data-math="J_\text{MultiTurn}(\pi)"></span>. This intuitively forms a closed-loop. The FM loss provides dense token-level supervision on every turn: predicting a critique like "you misidentified cell (2,2)" requires the model to internalize information about its own failure modes given its output, which sparse reward can never provide because it only signals success or failure. The multi-turn loss then rewards the model for acting on feedback effectively, teaching it what to do with failure-mode information.

This empirically works the best among all methods tested in this paper, which is a fascinating finding and the reason I rated this paper a 6. 

In practice, the feedback distribution <span class="math-render" data-display="inline" data-math="M(\cdot|x_h,y_h)"></span> that we would model would likely be human users, who would likely be given instructions on how to structure their feedback. However, in this paper they use a cleaner setting and set the feedback distribution as <span class="math-render" data-display="inline" data-math="p_\pi(c|x,y) = \pi_L(c|f_\text{FM}(x,y))"></span>, where <span class="math-render" data-display="inline" data-math="f_\text{FM}"></span> is a prompt template for eliciting feedback and <span class="math-render" data-display="inline" data-math="\pi_L"></span> is a larger model capable of providing high quality feedback.

They use the following cross entropy objective:

<span class="math-render" data-display="block" data-math="\ell_\text{FM}(\pi)=\mathbb{E}\left[ \sum_{h=0}^{H-1} -\log p_{\pi}(c_h|x_h,y_h)\right]"></span>

They also make a point about how it is unclear why feedback modeling would lead to improved performance, and then offer theoretical analysis about how it is a representation preconditioner. I will look deeper into these results later because the math is interesting regarding how they measure the extent to which sparse rewards update different representation directions and how the FM loss affects this. However, their analysis is on a log-linear model on a learned feature extractor, so the transfer from theory to practice is unclear, which they acknowledge.

Additionally, their specific statement is that "RLTF-FM trains the model to predict feedback, not to explicitly output a corrected answer, so its benefit is not obvious a priori". At this point they do not the attempt to explain the intuition of the FM loss alongside the multi-turn objective, and it is specifically when the two losses are put together that the intuition becomes clear to me. They only think more deeply about the two losses together afterwards in their theoretical analysis, which felt a bit odd. But it could very much be the case that I am missing something, because the authors were thorough throughout this paper as a whole.

## Interesting findings

SD and FM losses both generally work much better than GRPO (with some exceptions where the difference is small but still positive), and in general FM works better on reasoning and math while SD works better on creative writing.

The ablations show that meaningful/specific feedback greatly outperforms "you were correct/incorrect" when using the SD loss.

When using the FM loss, giving the model the feedback prompt <span class="math-render" data-display="inline" data-math="f_\text{FM}"></span> and having it model critiques at every turn at test time outperformed not using the self-critique strategy.

Really cool paper overall, I'm curious to see where it will lead.