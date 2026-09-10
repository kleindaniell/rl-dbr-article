Integrating Reinforcement Learning and the Theory of Constraints for Production Scheduling in Stochastic Manufacturing Environments

**Abstract:** Modern manufacturing systems often struggle to align demand with capacity, making production scheduling a critical yet challenging task --- especially under st ochastic conditions. Traditional heuristic and metaheuristic methods frequently yield suboptimal results due to their limited adaptability. Reinforcement Learning (RL), known for its ability to make sequential decisions in uncertain environments, has shown promise in overcoming these limitations and has outperformed conventional methods in various scheduling problems. While RL has been integrated with operations management theories such as Lean and Factory Physics, its combination with the Theory of Constraints (TOC) --- which focuses on managing system constraints under variability --- remains underexplored. This study investigates how integrating RL with TOC can enhance scheduling performance in stochastic manufacturing environments. We conducted two simulation experiments: one using a traditional TOC-based scheduling approach, and another combining RL with TOC. Results show that the RL-TOC hybrid solution achieved superior performance in terms of both production volume and inventory levels. This advantage is attributed to the RL agent's ability to adapt to system-wide work-in-process dynamics, beyond the constraint. We embedded this adaptive behavior into the traditional TOC scheduling solution, enhancing its drum-buffer-rope and buffer management components, which we believe is the main contribution of this study.

**Keywords:** theory of constraints; reinforcement learning; production scheduling; production planning and control; manufacturing

# 1 Introduction

The persistent lack of synchronization between demand and capacity continues to present significant challenges, even for modern manufacturing systems (Guo *et al.*, 2024). This mismatch often arises from the inherent variability within these systems, driven by factors such as market fluctuations, resource availability, and supply chain disruptions (Hopp and Spearman, 2021). Effective production scheduling plays a pivotal role in synchronizing demand with capacity by defining what to produce, how much to produce, where to produce, and in what sequence ([cf., Reference ≥ 2021]{.mark}). Traditionally, production scheduling has relied on various approaches, ranging from simple dispatching rules ([e.g., Reference]{.mark}) to more sophisticated algorithms ([e.g., Reference ≥ 2021]{.mark}). Recent advances in the field have increasingly incorporated meta-heuristics to achieve optimal or near-optimal solutions ([Reference ≥ 2021]{.mark}). However, using meta-heuristics in production scheduling presents challenges, particularly in dealing with the uncertainty and dynamic conditions typical of manufacturing environments ([Reference]{.mark}). Meta-heuristics such as genetic algorithms, simulated annealing, or tabu search are often designed for deterministic problems, where environmental conditions and decision outcomes are fixed ([cf. Reference]{.mark}). In contrast, stochastic optimization involves uncertainty and probabilistic outcomes, which meta-heuristics may struggle to accommodate, often leading to suboptimal solutions or long computation times ([Reference]{.mark}). These algorithms typically rely on heuristics that are not well-suited to random or fluctuating environments, especially when applied to large-scale or complex scheduling problems (Márquez and Ribeiro, 2022). Additionally, meta-heuristics may fail to fully capitalize on the stochastic nature of the problem, lacking the ability to learn from past experiences or adjust strategies based on observed randomness ([Reference]{.mark}). As a result, this can lead to inefficient exploration of the solution space and a higher likelihood of converging to a local optimum rather than a global one ([Reference]{.mark}). Furthermore, meta-heuristic algorithms are typically tailored to specific problems, making them less versatile and prone to poor generalization ([Reference]{.mark}). To overcome these challenges, reinforcement learning (RL) has emerged as a promising alternative for production scheduling ([Reference ≥ 2021]{.mark}). RL is a branch of machine learning where an agent interacts with its environment to maximize long-term rewards (Sutton and Barto, 2018). Its capacity to make sequential decisions and adapt to stochastic environments makes RL especially suitable for production scheduling applications ([cf., Li et al., 2022; del Real Torres et al., 2022; Wang et al., 2022]{.mark}).

RL has proven effective in various manufacturing contexts. For instance, Tang and Salonitis ([2021]{.mark}) leveraged deep RL with discrete-event simulation to optimize scheduling in reconfigurable systems, surpassing conventional dispatching rules. Chen et al. ([2022]{.mark}) addressed the flexible job shop scheduling problem with resource preemption using a deep RL-based algorithm, outperforming traditional meta-heuristics. Liu et al. ([2022]{.mark}) applied a Deep Q-Network algorithm for real-time scheduling in dynamic flexible job shop manufacturing. Gerpott et al. ([2022]{.mark}) used the Actor-Critic algorithm to minimize tardiness and makespan in a two-stage hybrid flow shop. Beyond standalone applications, RL has been integrated with some operations management (OM) theories. For example, Silva and Azevedo ([2019]{.mark}) applied RL to optimize work-in-process (WIP) caps in CONWIP and push systems, achieving a substantial reduction in WIP without sacrificing production volume. Xanthopoulos et al. ([2019]{.mark}) demonstrated RL's superiority in deriving optimal control policies for CONWIP-type systems. Paraschos et al. ([2023]{.mark}) used RL to optimize a three-stage production line, promoting Lean practices. Lahrichi et al. ([2023]{.mark}) used SARSA algorithm to dynamically parameterize demand-driven material requirements planning, reducing inventory and preventing stockouts.

One OM theory that has yet to be explored with RL in production scheduling is the Theory of Constraints (TOC). Like RL, which addresses decision-making in stochastic environments, TOC also does so, but with a strict focus on the constraint that limits the system's overall performance, usually protecting it through the use of buffers (Goldratt and Cox, 1984a). In production scheduling, TOC introduces two prominent techniques (Goldratt, 2006a; Schragenheim and Ronen, 1990): (i) drum-buffer-rope (DBR); and (ii) buffer management (BM). When applied together, these techniques, along with their associated performance measures --- throughput, inventory, and operating expenses --- enable the effective management of highly interdependent systems ([Reference]{.mark}). They achieve this by protecting capacity-constrained resources against shortages or blockages through inventory buffers while synchronizing inflows to the rhythm of the constraint ([Reference]{.mark}). TOC-based solutions to scheduling problems have demonstrated their effectiveness in various contexts. For instance, they have improved efficiency in production lines within aerospace engineering-to-order facilities (Telles *et al.*, 2022a) and increased productivity in small and medium-sized manufacturing enterprises (Buestán Benavides and Van Landeghem, 2015). Furthermore, these solutions have been successfully applied to automotive parts manufacturing (Golmohammadi, 2015), LCD panel production (Wu *et al.*, 2010), and engine cylinder restoration (Ma and Chen, 2009).

Despite the promising synergy between RL and TOC in production scheduling, their integration remains largely unexplored. This gap prompts the following research questions: (i) how can RL and TOC be combined to address production scheduling in stochastic manufacturing environments? (ii) what practical and theoretical outcomes can result from this combination? To address these questions, we employed computational simulation and conducted two experiments within the same stochastic manufacturing environment. In the first experiment, production scheduling was performed using a TOC-based solution (i.e., DBR and BM). In the second, scheduling was carried out combining RL with TOC. From a pragmatic perspective, a summative evaluation comparing these experiments demonstrated that a scheduling solution combining RL and TOC outperformed the TOC-based solution in terms of both production volume and inventory levels. Theoretically, a formative evaluation revealed that this advantage stems from the RL agent's ability to monitor and adapt to system-wide WIP dynamics, rather than focusing solely on the constraint. We embedded this adaptive behavior into the traditional TOC scheduling solution, enhancing its DBR and BM components, which we believe is the main contribution of this study.

The remainder of this paper is structured as follows. Section 2 provides the theoretical background on both RL and the TOC in the context of production scheduling. Section 3 outlines the research design, while Section 4 details the development of simulation models. Section 5 describes the setup of the simulation experiments, with the corresponding summative and formative evaluations presented in Section 6. Finally, Section 7 discusses the results, and Section 8 contextualizes the research contributions and limitations, highlighting opportunities for future studies.

# 2 Theoretical Background

# 2.1 Reinforcement Learning

RL is a branch of machine learning characterized by two core components ([Reference]{.mark}): (i) the agent; and (ii) the environment. The agent observes the current state of the environment, selects an action, and receives feedback in the form of a reward associated with that action (Sutton and Barto, 2018). Through iterative interactions with the environment, the agent refines its decision-making strategy, or policy, to maximize long-term rewards (Esteso *et al.*, 2023). Unlike traditional machine learning approaches that rely on static, pre-existing datasets, RL agents learn from dynamic, continuously evolving datasets generated through their ongoing interaction with a virtual environment (Panzer and Bender, 2022). This environment is often modeled using discrete-event simulation, which effectively captures system dynamics and variability (Kuhnle *et al.*, 2022; Nguyen *et al.*, 2022).

RL problems are typically framed as a sequence of interactions between the agent and the environment at discrete time steps $t \in \left\{ 0,\ 1,\ 2,\ \ldots,T \right\}$ ([Reference]{.mark}). At each time step $t$, the agent receives a representation of the environment's state, $S_{t}\mathcal{\in S}$, where $\mathcal{S}$ is the set of possible states ([Reference]{.mark}). Based on this state, the agent selects an action, $A_{t}\mathcal{\in A(}S_{t})$, where $\mathcal{A(}S_{t})$ represents the set of actions available in state $S_{t}$ ([Reference]{.mark}). One time step later, as a result of its action, the agent receives a numerical reward, $R_{t + 1}\mathcal{\in R \subset}\mathbb{R}$, and transitions to a new state, $S_{t + 1}$ ([Reference]{.mark}). In manufacturing applications of RL, the environment corresponds to the manufacturing system itself (or a model of it), such as an assembly line or warehouse ([e.g., Reference]{.mark}). The agent acts as the decision-maker, which could be a production scheduler, for example ([e.g., Reference]{.mark}). The state represents the current status of the system, such as the number of machines that are idle, running, or under maintenance ([e.g., Reference]{.mark}). The action refers to a decision the agent makes within the system, such as assigning a new work order to a machine or adjusting the priority of a pending job ([e.g., Reference]{.mark}). The reward represents the payoff the agent receives in response to its actions, typically aimed at improving a specific performance measure, such as production volume or lead time ([e.g., Reference]{.mark}).

The environment in RL can be deterministic or stochastic. In the latter case, it is often modeled as a finite Markov Decision Process (MDP), where the environment's response at $t + 1$ depends only on the state and action representations at $t$ ([Reference]{.mark}). This assumption simplifies the dynamics of agent-environment interactions, which can be described by a distribution function, such as ([Reference]{.mark}):

  ------------------------------------------------------------------------------------------------------------------------------------------
           $p\left( s',r \middle| s,a \right) = \Pr\left\{ S_{t + 1} = s',\ R_{t + 1} = r \middle| S_{t} = s,A_{t} = a \right\}$,   \(1\)
  -------- ------------------------------------------------------------------------------------------------------------------------ --------

  ------------------------------------------------------------------------------------------------------------------------------------------

for all possible states $s$, $s'$, rewards $r$, and actions $a$. This formulation allows prediction of the next state and expected reward based on the current state and action, though it requires prior knowledge on the distribution function ([Reference]{.mark}).

The agent's behavior is governed by a policy, $\pi_{t}\left( a \middle| s \right)$, which represents the probability of taking action $A_{t} = a$ in state $S_{t} = s$ ([Reference]{.mark}). If the policy involves no randomness, then only one action can be taken from each state, resulting in a deterministic policy $\pi_{t}(s)$ ([Reference]{.mark}). RL methods define how the agent updates its policy based on its experiences, with the ultimate goal of maximizing the total rewards received over time, known as the return $G_{t}$:

  ----------------------------------------------------------------------------------
           $$G_{t} = \sum_{k = 0}^{T - t - 1}{\gamma^{k}R_{t + k + 1},}$$   \(2\)
  -------- ---------------------------------------------------------------- --------

  ----------------------------------------------------------------------------------

where $\gamma$ is the discount rate $(0 \leq \gamma \leq 1)$ that determines the present value of future rewards ([Reference]{.mark}). A reward received $k$ time steps in the future is worth only $\gamma^{k - 1}$ times its immediate value.

Most RL algorithms rely on estimating value functions, which assess the 'goodness' of being in a given state or taking a specific action ([Reference]{.mark}). This goodness is defined in terms of the expected return $\mathbb{E}_{\pi}\lbrack.\rbrack$ and is essential for identifying optimal policies $\pi_{*}$, where $\upsilon_{*}(s) \geq \upsilon_{\pi}(s)$ or $q_{*}(s,a) \geq q_{\pi}(s,a)$ for all $s \in \mathcal{S}$ and $a \in \mathcal{A}$, ([cf., Reference]{.mark}). The two primary value functions for finite MDPs are ([Reference]{.mark}):

i.  State-value function $(\upsilon_{\pi}(s))$: The expected return starting in state $s$ and following policy $\pi$:

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
           $$\upsilon_{\pi}(s) = \mathbb{E}_{\pi}\left\lbrack G_{t}|S_{t} = s \right\rbrack = \mathbb{E}_{\pi}\left\lbrack \left. \ \sum_{k = 0}^{T - t - 1}{\gamma^{k}R_{t + k + 1}} \right|S_{t} = s \right\rbrack;$$   \(3\)
  -------- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --------

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

ii. Action-value function $(q_{\pi}(s,a))$: The expected return starting from state $s$, taking action $a$, and subsequently following policy $\pi$:

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        $$q_{\pi}(s,a) = \mathbb{E}_{\pi}\left\lbrack G_{t}|S_{t} = s,A_{t} = a\  \right\rbrack = \mathbb{E}_{\pi}\left\lbrack \left. \ \sum_{k = 0}^{T - t - 1}{\gamma^{k}R_{t + k + 1}} \right|S_{t} = s,\ A_{t} = a \right\rbrack.$$   \(4\)
  ----- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --------

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Value functions satisfy recursive relationships, which allows the use of Bellman optimality equation ([cf., Reference]{.mark}). This leads to the optimal forms of the state-value function ${(\upsilon}_{*}(s))$ and the action-value function ${(q}_{*}(s,a)$), defined as follows ([Reference]{.mark}):

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------
        $$\upsilon_{*}(s) = \max_{a \in A(s)}{\sum_{s',r}^{}{p\left( s',r \middle| s,a \right)\left\lbrack r + \gamma\upsilon_{*}(s') \right\rbrack}.}$$   \(5\)
  ----- -------------------------------------------------------------------------------------------------------------------------------------------------- --------
        $$q_{*}(s,a) = \sum_{s',r}^{}{p\left( s',r \middle| s,a \right)}\left\lbrack r + \gamma\max_{a'}{q_{*}(s',a')} \right\rbrack.$$                    \(6\)

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------

The computation of optimal policies depends on how the agent predicts the environment\'s response to its actions ([Reference]{.mark}). In model-based methods, where the agent uses a model to plan actions in advance, such as estimating $p\left( s',r \middle| s,a \right)$, dynamic programming algorithms are typically employed ([e.g., Reference]{.mark}). In contrast, model-free methods focus on learning the direct relationship between actions and returns through 'trial and error', often utilizing temporal-difference algorithms ([Reference]{.mark}). Model-free methods can be further divided into on-policy and off-policy strategies ([Reference]{.mark}). An on-policy strategy learns from the actions taken by the current policy, with SARSA being the prominent learning algorithm ([e.g., Reference]{.mark}), while an off-policy strategy learns from actions taken by a combination of the current and different policies, typically implemented using the Q-learning algorithm ([e.g., Reference]{.mark}). Furthermore, model-free methods can be categorized into gradient-based and gradient-free methods. Gradient-based methods, such as Proximal Policy Optimization (PPO), optimize policies by directly adjusting parameters in the direction of the gradient of the expected return with respect to the policy parameters ([Reference]{.mark}). The PPO algorithm specifically improves upon traditional policy gradient methods by employing a clipped objective function to prevent large updates that could destabilize training ([Reference]{.mark}). The core PPO objective is expressed as follows:

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        $$L^{CLIP}(\theta) = \mathbb{E}_{t}\left\lbrack \min\left( p_{t}(\theta){\widehat{A}}_{t},\ clip(p_{t}(\theta),1 - \epsilon,1 + \epsilon){\widehat{A}}_{t} \right) \right\rbrack;$$   \(7\)
  ----- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --------

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

where $p_{t}(\theta) = \frac{\pi_{\theta}\left( a|s \right)}{\pi_{\theta_{old}}\left( a|s \right)}$ is the probability ratio, ${\widehat{A}}_{t} = G_{t} - \upsilon_{\pi}(s)$ is the estimated advantage, and $\epsilon$ is a small hyperparameter controlling the clipping range. This method ensures that the updates are stable and avoid excessively large policy changes ([Reference]{.mark}). Gradient-free methods, on the other hand, typically rely on search-based techniques or evolutionary strategies to update the policy without directly computing gradients ([Reference]{.mark}). These methods are particularly useful in environments where the gradient is difficult to compute or highly noisy ([Reference]{.mark}). More recently, both model-based and model-free methods have incorporated deep learning, leading to algorithms like Actor-Critic and Deep Q-Network ([e.g., Reference 1, Reference 2]{.mark}), which have paved the way for a new branch of RL known as deep RL ([cf., Reference]{.mark}). The ability to make autonomous sequential decisions and adapt to stochastic environments makes RL particularly well-suited for production scheduling applications ([cf., Reference 1, Reference 2]{.mark}), providing a foundation for the remainder of this work.

# 2.2 Theory of Constraints

A constraint is any factor that prevents a system from achieving its goal (Mabin and Balderstone, 2003). In manufacturing environments, constraints can take the form of equipment with limited capacity or low reliability, operations that produce high rates of defective parts, or policies that limit the system's flexibility or efficiency (Gupta and Boyd, 2008). Addressing the system's constraint often involves five major steps (Watson *et al.*, 2007): (i) identify the system's constraint; (ii) exploit the constraint; (iii) subordinate everything else to the constraint; (iv) elevate the constraint; and (v) if the constraint is resolved, restart the process from step (i).

When demand $d$ exceeds the planned capacity $p$, identifying the system's constraint (step 1) requires locating the resource $j \in \left\{ 0,\ 1,\ 2,\ \ldots,J \right\}$ with the highest load $l_{j}$, defined as ${argmax}_{j}l_{j}$ ([Reference]{.mark}). In a stochastic manufacturing environment producing multiple products $i \in \left\{ 0,\ 1,\ 2,\ \ldots,I \right\}$ in quantities $m_{i}$, utilizing $J$ resources with expected cycle times $\mathbb{E}\left\lbrack \mathcal{T}_{ij}^{c} \right\rbrack$​, considering each resource's expected overall effectiveness $\mathbb{E}\left\lbrack \mathcal{E}_{j} \right\rbrack$​​ and change-over time $\mathbb{E}\left\lbrack \mathcal{T}_{j}^{CT} \right\rbrack$, the load on a resource can be calculated as follows ([Reference]{.mark}):

+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+
| $$d = \left. \ \sum_{\forall i}^{}{m_{i}\max\left( \frac{\mathbb{E}\left\lbrack \mathcal{T}_{ij}^{c} \right\rbrack}{\mathbb{E}\left\lbrack \mathcal{E}_{j} \right\rbrack} + \frac{\mathbb{E}\left\lbrack \mathcal{T}_{j}^{CT} \right\rbrack}{m_{i}} \right)} \right|D_{i} \in D \land D = d;$$                                                                                                                                                           | \(8\)  |
+:===============================================================================================================================================:+:======================================================================================================================================================================================================================================================================================================:+========+
|                                                                                                                                                 | $$\left. \ l_{j} = \frac{1}{p}\left\lbrack \left( \sum_{\forall i}^{}{m_{i}\frac{\mathbb{E}\left\lbrack \mathcal{T}_{ij}^{c} \right\rbrack}{\mathbb{E}\left\lbrack \mathcal{E}_{j} \right\rbrack}} \right)\mathbb{+ E}\left\lbrack \mathcal{T}_{j}^{CT} \right\rbrack \right\rbrack \right|d \geq p.$$ | \(9\)  |
+-------------------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+

Once the constraint is identified, the next step is to determine the most effective way to exploit it (step 2) and then adjust all other operations accordingly (step 3) (Cox and Schleier, 2010). In production scheduling, this is achieved by implementing DBR and BM (Telles *et al.*, 2020a). The constraint corresponds to the drum in DBR, which dictates the overall system's performance (Darlington *et al.*, 2015). In TOC, manufacturing performance is primarily measured by throughput $(\tau)$, which represents the money generated from sales (Stefano *et al.*, 2022). Throughput is calculated as the difference between revenue $(\rho)$ and total variable costs $(c)$ for each product $i$, multiplied by its respective quantities produced and sold. This can be expressed as:

  -----------------------------------------------------------------------------------
        $$\tau = \sum_{i = 1}^{I}{\left( \rho_{i} - c_{i} \right)m_{i}}.$$   \(10\)
  ----- -------------------------------------------------------------------- --------

  -----------------------------------------------------------------------------------

Additionally, there are two other performance measures ([Reference]{.mark}): (i) inventory $\mathcal{(i)}$, which represents the total money invested in purchasing items intended for sale; and (ii) operating expenses $(e)$, which reflect the money spent to convert inventory into throughput.

Buffers are another class of element in DBR, typically categorized into two types (when there are no assembly stations). The first is the constraint buffer (CB), located before the constraint to protect its flow and ensure it operates without shortages ([Reference]{.mark}). The second is the shipping buffer (SB), positioned at the end of the system to protect against disruptions to the flow at any point in the process ([Reference]{.mark}). The initial size of these buffers $\left( b_{i}^{x} \right)$ for each product $i$ is typically defined by the quantity of semi-finished or finished products corresponding to half the expected lead time from the start of the process $j = 1$ to the position $j = n$, where the buffer $\mathcal{x} \in \left\{ CB,SB \right\}$ is located ([cf., Reference]{.mark}). The lead time $\mathcal{T}_{i}^{\mathcal{l}}$ is a random variable resulting from the sum of all cycle times $\mathcal{T}_{ij}^{c}$ and waiting times $\mathcal{T}_{ij}^{w}$ belonging to a specific production route $1 \rightarrow n$, and its expected value is defined as:

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        $$\mathbb{E}\left\lbrack \mathcal{T}_{i}^{\mathcal{l}}\mathcal{|x} \right\rbrack\  = \ \sum_{\forall j \in (1 \rightarrow n)}^{}{\mathbb{E}\left\lbrack \mathcal{T}_{ij}^{c} \right\rbrack\mathbb{+ E}\left\lbrack \mathcal{T}_{ij}^{w} \right\rbrack}\left\{ \binom{n = {argmax}_{j}l_{j}\ }{n = N\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ }\binom{if\mathcal{\ x =}CB}{if\mathcal{\ x =}SB} \right.\ .$$   \(11\)
  ----- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --------

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

According to Little's Law ([Reference]{.mark}), the initial buffer size can be estimated as follows:

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        $$b_{i}^{\mathcal{x}} \approx \min\left( \frac{\mathbb{E}\left\lbrack \mathcal{E}_{j} \right\rbrack}{\mathbb{E}\left\lbrack \mathcal{T}_{ij}^{c} \right\rbrack} \right)\left( \frac{\mathbb{E}\left\lbrack \mathcal{T}_{i}^{\mathcal{l}} \right\rbrack}{2} \right).$$   \(12\)
  ----- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --------

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The initial buffer size is often dynamically adjusted to a 'steady state' reference level ${b'}_{i}^{x}$ using Buffer Management (BM) policies (Schragenheim, 2010a). Within the TOC framework, the buffers are typically divided into three distinct zones---red, yellow, and green. Each zone represents one-third of the current reference level, with the red zone indicating the lowest inventory level and the green zone the highest (Telles *et al.*, 2022b). The reference leve is continuous evaluated over a replenishment time $\mathcal{T}_{i}^{R}$, that is difined as the sum of order planning interval $\mathrm{\Delta}\varphi$ and the lead time $\mathcal{T}_{i}^{\mathcal{l}}$ (Schragenheim, 2010b). If the actual buffer level, $H_{i}^{x}$, remains consistently within the green zone for a duration of at least three replenishment times $3\mathcal{T}_{i}^{R}$, the reference level is reduced by 15% for the subsequent order planning interval, $\mathrm{\Delta}\varphi + 1$. The reduction logic is defined as:

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        $${b'}_{i,\Delta\varphi + 1}^{x} = 0.85b_{i,\Delta\varphi}^{x}\quad\text{if}\quad H_{i,t}^{x} \geq \frac{2}{3}b_{i,\Delta\varphi}^{x}\ |\ \forall t \in \left\lbrack 0,\ 3\mathcal{T}_{i}^{R} \right\rbrack$$   \(13\)
  ----- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --------

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Conversely, the buffer must be expanded if the cumulative penetrations into the red zone, $k_{i}^{x}$, achieve or exceed the total size of the red zone during a single replenishment time $\mathcal{T}_{i}^{R}$ (Schragenheim, 2010b). The reference level is increased by 20% for the next order planning interval $\Delta\varphi$. The expansion logic is defined as:

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        $${b'}_{i,\Delta\varphi + 1}^{x} = 1.2b_{i,\Delta\varphi}^{x}\quad\text{if}\quad\sum_{t = 0}^{\mathcal{T}_{i}^{R}}k_{i,t}^{x} \geq \frac{1}{3}b_{i,\Delta\varphi}^{x}$$   \(14\)
  ----- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- --------

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The rationale for these adjustments is to protect the production flow while maintaining the minimum inventory $\mathcal{i}$.

Similar to buffers, production scheduling is another approach to exploiting the constraint (step 2) ([Reference]{.mark}). In TOC, the goal of scheduling is to maximize throughput during the replenishment interval ([Reference]{.mark}). Therefore, the sequence of products $i$ to be produced is defined in descending order of throughput, moderated by the priority level of finished products in the SB, i.e., ${\left( \frac{\mathcal{b'}_{i}^{SB}}{H_{i}^{SB}} \right)\tau}_{i}^{(1)} \geq \left( \frac{\mathcal{b'}_{i}^{SB}}{H_{i}^{SB}} \right)\tau_{i}^{(2)} \geq \ldots \geq \left( \frac{\mathcal{b'}_{i}^{SB}}{H_{i}^{SB}} \right)\tau_{i}^{(\mathfrak{n})}$, based on the following equation:

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------
   $$\left. \ \underset{m_{i}}{argmax\ }\tau \right|\frac{\mathrm{\Delta}d}{\mathrm{\Delta}\varphi} \leq \frac{\mathrm{\Delta}p}{\mathrm{\Delta}\varphi}.$$  \(15\)
  ---------------------------------------------------------------------------------------------------------------------------------------------------------- ---------

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------

The rope is the final element of DBR, and its function is to subordinate the rest of the system to the constraint (step 3) ([Reference]{.mark}). Often, the demand is greater than the capacity during the replenishment interval, which could lead to the creation and release of orders that the system cannot fully execute, ultimately increasing $\mathcal{i}$ ([Reference]{.mark}). Therefore, the rope's role is to release production orders based on the consumption of the CB, as outlined in [[Figure 3](#_Ref187516903)]{.mark}.

The quantity of each product $i$ to be produced during the replenishment interval $M_{i}^{\mathrm{\Delta}\varphi}$ is typically a function of the SB level $H_{i}^{SB}$ and its reference $\mathcal{b'}_{i}^{SB}$ and the work-in-process $W_{i}^{SB}$ along a complete production route $1 \rightarrow N$, as described by the following equations (Cox III and Schleier Jr., 2010):

  --------------------------------------------------------------------------------------------------------------
                   $$W_{i}^{SB} = \sum_{\forall j \in (1 \rightarrow N)}^{}w_{ij};$$                   \(16\)
  ---------------------------------------------------------------------------------------------------- ---------
   $M_{i}^{\mathrm{\Delta}\varphi} = \left( \mathcal{b'}_{i}^{SB} - H_{i}^{SB} \right) - W_{i}^{SB};$  \(17\)

  --------------------------------------------------------------------------------------------------------------

where $w_{ij}$ represents the quantity of product $i$ being produced at the resource $j$. This quantity is stored in a 'virtual buffer' known as the order buffer (OB), which typically has an initial infinite size $\left( \mathcal{b}_{i}^{OB} = \infty \right)$ and a reference level based on the replenishment interval, i.e., ${\mathcal{b}'}_{i}^{OB} = M_{i}^{\mathrm{\Delta}\varphi}$. Additionally, the quantity of each product $i$ to be released for production within a given time interval $M_{i}^{\mathrm{\Delta}t}$ is a function of the CB level $H_{i}^{CB}$, its reference $\mathcal{b'}_{i}^{CB}$, and the work-in-process $W_{i}^{CB}$ along a partial production route $1 \rightarrow {argmax}_{j}l_{j}$ that ends at the constraint. This relationship is described as follows:

  ----------------------------------------------------------------------------------------------------------
   $$W_{i}^{CB} = \sum_{\forall j \in \left( 1 \rightarrow {argmax}_{j}l_{j} \right)}^{}w_{ij};$$  \(18\)
  ------------------------------------------------------------------------------------------------ ---------
   $$M_{i}^{\mathrm{\Delta}t} = \left( \mathcal{b'}_{i}^{CB} - H_{i}^{CB} \right) - W_{i}^{CB}.$$  \(19\)

  ----------------------------------------------------------------------------------------------------------

The number of orders released reduces the current level of the OB, denoted as $H_{i}^{OB}$.

Step 4 may involve increasing shifts, hiring additional labor, or purchasing new equipment to increase the capacity of the constraint (Goldratt, 2006b). Step 5 continues the cycle of continuous improvement within TOC (Goldratt and Cox, 1984b). Both of these steps are beyond the scope of the current work.

# 3 Research Design

This study employed computational simulation ([Law, 2015]{.mark}) to explore the integration of RL and TOC in addressing production scheduling within stochastic manufacturing environments, along with the associated practical and theoretical implications ([cf., Harrison et al. 2007; Davis et al. 2007]{.mark}). The research followed a structured five-stage process, as illustrated in [[Figure 1](#_Ref187516847)]{.mark}: (i) construction of the virtual environment; (ii) development of the simulation model based on the TOC solution; (iii) creation of the simulation model integrating RL and TOC; (iv) execution of simulation experiments; and (v) summative and formative evaluation.

![](overleaf/figures/media/image1.png){width="4.081512467191601in" height="5.131220472440945in"}

[]{#_Ref187516847 .anchor}[Figure 1.]{.mark} [Overview of the research design.]{.mark}

In stage 1, the virtual environment was constructed as the foundation for creating the two subsequent simulation models in stages 2 and 3. The process began with selecting a reference model in step 1.1. The discrete-event simulation model proposed by (Atwater and Chakravorty, 2002a) was chosen for its ability to simulate a stochastic manufacturing environment with multiple products, resources, and production routes (see [[Figure 2](#_Ref187516873)]{.mark}). In step 1.2, we used this reference model to implement the virtual environment computationally, employing Python along with the SimPy framework for discrete-event simulation ([Reference]{.mark}) and OpenAI Gym, a toolkit for developing and comparing RL algorithms ([Reference]{.mark}). Finally, in step 1.3, we introduced additional probabilistic elements to the validated model to increase the variance of the virtual environment ([cf., Machado et al., 2023]{.mark}). These elements included resource changeovers and downtime, such as time between failures and time to repair.

In stage 2, the virtual environment was used to develop a simulation model that incorporated a TOC-based scheduling solution, combining DBR and BM. First, an operational assessment of the virtual environment was conducted, which led to identifying the system's constraint in step 2.1. In step 2.2, once the constraint (i.e., the drum) was defined, the remaining two elements of DBR --- the buffer and the rope --- were implemented following (Schragenheim, 2010c). Two types of buffers were considered (see [[Figure 3](#_Ref187516903)]{.mark}): (i) the CB and (ii) the SB. The rope was implemented as a logic for creating, scheduling, and releasing orders and raw materials at the system's entrance, based on the consumption of both buffers. A BM policy was then introduced and applied to optimize the reference levels of the shipping buffers in steps 2.3 (Schragenheim, 2010c). In step 2.4, an experiment was conducted to identify the best reference level for CB.

As in stage 2, the virtual environment informed the development of the simulation model that integrated RL and TOC into a production scheduling solution in stage 3. In step 3.1, fifty-two state variables were considered, capturing the current quantity of each product being processed throughout the system, including in the CB and SB. With respect to the agent's actions, in step 3.2 we the decisions as how much of each product to produce during the replenishment interval. The scheduling and release of orders followed the logic established by TOC (see [[Figure 5](#_Ref187516938)]{.mark}). In step 3.3, the reward function was modeled by incorporating four penalty subfunctions aligned with TOC's two key performance measures: (i) throughput and (ii) inventory. Finally, in step 3.4, the PPO algorithm was employed to train the agent in its search for the optimal policy ([cf. Reference]{.mark}). This training process was carried out across 100,000 interactions and implemented using the Stable Baselines3 library ([Raffin et al., 2021]{.mark}).

In stage 4, both simulation models --- one based on TOC (step 4.1) and the other integrating RL with TOC (step 4.2) --- were executed over multiple replications (or episodes), simulating 200.000 time units of operation. The performance measures used as model outputs included: (i) the number of products produced and sold; (ii) the average number of parts in process; (iii) the average lead-time; and (iv) the average inventory, all measured over a predefined replenishment interval. In step 4.3, these measures were recorded both by individual product and in aggregate, resulting in a database generated from the simulation runs.

In the final stage, the database of simulation runs underwent both summative and formative evaluations ([cf., Venable, 2016]{.mark}). The summative evaluation (step 5.1) aimed to quantify the performance difference between the two simulation models, using the TOC-based model as a reference. This analysis employed a confidence interval for the paired mean difference ([cf., Law, 2015 p. 560]{.mark}). The formative evaluation (step 5.2) focused on understanding and explaining the underlying reasons for the observed performance differences. This process unfolded in two phases. First, three analytical techniques were applied --- Spearman's coefficient ([Yu et al., 2021]{.mark}), multiple linear regression ([Nathans et al., 2012]{.mark}), and decision trees ([Kuhnle et al., 2022]{.mark}) --- to explore the relationship between state variables and the actions taken in each model. Based on the insights drawn from these analyses, mechanism-based explanations were formulated to articulate the dynamic hypotheses underlying the key differences ([cf., Gauss et al. 2024a]{.mark}). This also highlighted opportunities for improving traditional DBR. Finally, in step 4.3, a new TOC-based simulation model incorporating the modified DBR approach was built and again compared with the model integrating RL and TOC. Further details about the virtual environment and simulation models are provided in section 4, while sections 5 and 6 present the simulation runs and the summative and formative evaluations, respectively.

# 4 Simulation Models

# 4.1 Virtual Environment 

The reference model by Atwater and Chakravorty ([2002]{.mark}), which serves as the basis for constructing the virtual environment, represents a job-shop manufacturing system that operates 24 hour per day, seven days a week, producing ten different products. The demand and production routes for these products are outlined in [[Figure 2](#_Ref187516873)]{.mark} and detailed in Table S1 of the supplementary material. Each product requires [one]{.mark} unit of raw material for production. The system utilizes 13 resources, with infinite queue in front of each one, and cycle times that vary depending on the specific product being produced; these cycle times are recorded in Table S2. The reference model assumes no changeovers or downtime for the resources. Therefore, considering $\mathbb{E}\left\lbrack \mathcal{T}_{j}^{CT} \right\rbrack = 0$, $\mathbb{E}\left\lbrack \mathcal{E}_{\forall j} \right\rbrack = 1$​​, and the average demands from Table S1 (denoted as $m_{i}$), the primary constraint in the system is resource nine, as determined by Equation 9 ([cf., Reference]{.mark}).

The virtual environment was implemented computationally using Python 3.11, with one time step representing one hour. Due to its stochastic nature, the SimPy framework for discrete-event simulation (Eriksson *et al.*, 2022) was employed. Additionally, OpenAI Gym was used to facilitate its subsequent integration with RL algorithms (Brockman *et al.*, 2016; Lang *et al.*, 2021). The reference model consists of a terminal system than rans for 200,000 hour (Atwater and Chakravorty, 2002b).

To increase the variance of the virtual environment, we introduced changeovers and downtime to the resources. The changeover time $\left( \mathcal{T}_{j}^{CT} \right)$ was modeled using a Normal distribution, as this is typically applied when the operation consists of a series of independent and sequential tasks (Yu and Ram, 2006). The average changeover time was randomly assigned for each resource using a Uniform distribution, $U(0.0,\ 0.5)$. Similarly, the standard deviation of the changeover time was defined using a Uniform distribution, $U(0.0,\ 0.3)$. Regarding downtime, it was applied to all resources through distributions for the time between failures and time to repair. Time between failures $\left( \mathcal{T}_{j}^{TBF} \right)$ was modeled using an Exponential distribution, assuming that all resources are in the maturity phase of their life cycle and thus exhibit a constant failure rate (Sellitto and Brusius, 2017). The failure rate was randomly defined using a Uniform distribution, $U(0.001,\ 0.04)$. For time to repair $\left( \mathcal{T}_{j}^{TTR} \right)$, similar to changeover time, the distribution was modeled using a Normal distribution, with the mean and standard deviation defined by the Uniform distributions $U(5.0,\ 15.0)$ and $U(1.0,\ 3.0)$, respectively. []{.mark}[Table 1](#_Ref187517135) summarizes the values used for each resource.

+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| []{#_Ref187517135 .anchor}Table 1. Elements of variance introduced into the virtual environment.                                                                                                                                                                                                                                                                                                                                                        |
+-----------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------+
| Resource                                | $\mathcal{T}_{j}^{CT}$^(1)^                                                                                                 | $\mathcal{T}_{j}^{TBF}$^(1)^                                                      | $\mathcal{T}_{j}^{TTR}$^(1)^                                                                                                | $\mathbb{E}\left\lbrack \mathcal{E}_{j} \right\rbrack$^(2,3)^ |
+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+                                                               |
| $$j$$                                   | Distribution                            | Mean                                    | Std. dev.                               | Distribution                            | Failure rate                            | Distribution                            | Mean                                    | Std. dev.                               |                                                               |
+=========================================+=========================================+=========================================+=========================================+=========================================+=========================================+=========================================+=========================================+=========================================+===============================================================+
| 1                                       | Normal                                  | 0.25                                    | 0.09                                    | Exponential                             | 0.0098                                  | Normal                                  | 5.24                                    | 2.97                                    | 0.89                                                          |
+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------------------------+
| 2                                       | Normal                                  | 0.04                                    | 0.05                                    | Exponential                             | 0.0058                                  | Normal                                  | 5.95                                    | 1.94                                    | 0.93                                                          |
+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------------------------+
| 3                                       | Normal                                  | 0.42                                    | 0.17                                    | Exponential                             | 0.0116                                  | Normal                                  | 5.28                                    | 1.91                                    | 0.86                                                          |
+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------------------------+
| 4                                       | Normal                                  | 0.41                                    | 0.22                                    | Exponential                             | 0.0162                                  | Normal                                  | 2.35                                    | 1.16                                    | 0.88                                                          |
+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------------------------+
| 5                                       | Normal                                  | 0.34                                    | 0.10                                    | Exponential                             | 0.0252                                  | Normal                                  | 6.00                                    | 2.16                                    | 0.88                                                          |
+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------------------------+
| 6                                       | Normal                                  | 0.49                                    | 0.13                                    | Exponential                             | 0.0376                                  | Normal                                  | 5.14                                    | 2.22                                    | 0.83                                                          |
+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------------------------+
| 7                                       | Normal                                  | 0.16                                    | 0.09                                    | Exponential                             | 0.0079                                  | Normal                                  | 5.31                                    | 1.95                                    | 0.92                                                          |
+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------------------------+
| 8                                       | Normal                                  | 0.27                                    | 0.21                                    | Exponential                             | 0.0104                                  | Normal                                  | 4.45                                    | 1.93                                    | 0.90                                                          |
+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------------------------+
| 9                                       | Normal                                  | 0.45                                    | 0.23                                    | Exponential                             | 0.0045                                  | Normal                                  | 2.69                                    | 1.21                                    | 0.95                                                          |
+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------------------------+
| 10                                      | Normal                                  | 0.49                                    | 0.12                                    | Exponential                             | 0.0188                                  | Normal                                  | 5.01                                    | 2.85                                    | 0.82                                                          |
+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------------------------+
| 11                                      | Normal                                  | 0.39                                    | 0.17                                    | Exponential                             | 0.0301                                  | Normal                                  | 3.15                                    | 2.39                                    | 0.85                                                          |
+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------------------------+
| 12                                      | Normal                                  | 0.25                                    | 0.05                                    | Exponential                             | 0.0194                                  | Normal                                  | 5.59                                    | 1.50                                    | 0.87                                                          |
+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------------------------+
| 13                                      | Normal                                  | 0.28                                    | 0.07                                    | Exponential                             | 0.0171                                  | Normal                                  | 2.95                                    | 1.71                                    | 0.89                                                          |
+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+-----------------------------------------+---------------------------------------------------------------+
| \(1\) Unit of time: hour; (2) Unit of Effectiveness: percentage; (3) $\mathbb{E}\left\lbrack \mathcal{E}_{j} \right\rbrack = \frac{\mathbb{E}\left\lbrack \mathcal{T}_{j}^{TBF} \right\rbrack}{\left( \mathbb{E}\left\lbrack \mathcal{T}_{j}^{TBF} \right\rbrack\mathbb{+ E}\left\lbrack \mathcal{T}_{j}^{TTR} \right\rbrack \right)}$.                                                                                                                 |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

# 4.2 TOC-Based Simulation Model

The first step in developing the TOC-based simulation model was to identify the constraint, or the drum of the DBR ([cf., Reference]{.mark}). Using the data from []{.mark}[Table 1](#_Ref187517135), along with Tables S1 and S2 in the supplementary material, and assuming $p = \mathrm{\Delta}\varphi = 168$ hours, it was confirmed that resource nine was the system's constraint, as it had the highest load (i.e., $l_{9} = 0.99$), according to Equation 9. Consequently, the positions of the CB and SB was set, as outlined in [[Figure 2](#_Ref187516873)]{.mark}.

![](overleaf/figures/media/image2.png){width="4.640970034995625in" height="4.539130577427821in"}

[]{#_Ref187516873 .anchor}[Figure 2. Outline of the reference model by Atwater and Chakravorty (2002).]{.mark}

The implementation of the DBR rope considered three aspects, as illustrated in [[Figure 3](#_Ref187516903)]{.mark}: (i) order creation; (ii) scheduling; and (iii) order release. Order creation takes place at the start of the order planning interval $\mathrm{\Delta}\varphi$, which in this case is [168 hours]{.mark}. The order size $\left( M_{i}^{\mathrm{\Delta}\varphi} \right)$ is calculated by Equation 17 and determines the reference level of the OB for the order planning interval, i.e., $\mathcal{b'}_{i}^{OB} = M_{i}^{\mathrm{\Delta}\varphi}$. Order scheduling is modeled by Equation 15, which considers the throughput $\tau_{i}$ and the priority level of finished products in the SB $\left( \frac{\mathcal{b'}_{i}^{SB}}{H_{i}^{SB}} \right)$ ([cf., Reference]{.mark}). In this context, we assume $\left( \rho_{i} - c_{i} \right) = 1$ when defining $\tau_{i}$ in the Equation 10, as the original work by Atwater and Chakravorty ([2002]{.mark}) does not consider the revenue and variable cost of each product. Finally, order release, represented by $M_{i}^{\mathrm{\Delta}t}$, considers the consumption of the CB $\left( \mathcal{b'}_{i}^{CB} - H_{i}^{CB} \right)$ and the work-in-process at each resource up to the constraint $W_{i}^{CB}$ ([cf., Reference]{.mark}), as defined in Equation 19. The number of orders released moderates the current level of the OB, denoted as $H_{i}^{OB}$.

![Diagrama Descrição gerada automaticamente](overleaf/figures/media/image3.png){width="6.295138888888889in" height="4.393409886264217in"}

[]{#_Ref187516903 .anchor}[Figure 3. Outline of the TOC-based simulation model.]{.mark}

The reference levels of the shipping buffers (SB) were calibrated with the buffer-management (BM) policy described in Section 2.2. Fifteen independent replications were run, each for 5,000,000 time units, starting from a common oversized SB target. BM adjustments were held off for the first 100,000 time units and then applied. For each product, the SB target was aligned on a common time grid and averaged across replications (Figure 4). The replication-average trajectories were inspected graphically to identify the onset of steady state (Law, 2015). From time 2,000,000 onward, the mean target of each replication was computed and then averaged over the 15 replications; those values, after rounding, were used as the fixed SB reference levels in the subsequent experiments.

![](overleaf/figures/media/image4.png){width="6.295138888888889in" height="3.1222222222222222in"}

[Figure 4. Optimization of shipping buffer (SB) reference levels.]{.mark}

The final reference level for each product in each buffer $\left( \mathcal{b'}_{i}^{x} \right)$, measured in terms of quantity of semi-finished or finished products, is presented in [Table 2](#_Ref187517181). These reference levels remained constant throughout the experiments conducted in Section 5.

Table 2. Final reference leves for the shipping buffer (SB)

  -------------------------------------------------
  Product $i$           $$\mathcal{b'}_{i}^{SB}$$
  --------------------- ---------------------------
  1                     [0]{.mark}

  2                     [0]{.mark}

  3                     [0]{.mark}

  4                     [0]{.mark}

  5                     [0]{.mark}

  6                     [0]{.mark}

  7                     [0]{.mark}

  8                     [0]{.mark}

  9                     [0]{.mark}

  10                    [0]{.mark}
  -------------------------------------------------

The constraint-buffer (CB) target level was defined from an experiment with 12 levels (from 250 to 800 units, in 50-unit steps), each simulated for 50 replications. The decision used metrics with adequate sample size (95% CI relative half-width ≤ 5%) that also varied significantly with CB size (Kruskal-Wallis, α = 0.05). The level 350 was selected because it sustained the same throughput as larger buffers while requiring the smallest inventory, as presented on table 3.

[]{#_Ref187517181 .anchor}Table 2. Metrics comparison between constraint buffer (CB) target level.

  ---------------------------------------------------------------
  $$\mathcal{b'}_{i}^{CB}$$   $$\tau$$           $$i$$
  --------------------------- ------------------ ----------------
  250                         [97,861]{.mark}    [1,993]{.mark}

  300                         [99,884]{.mark}    [2,082]{.mark}

  350                         [100,032]{.mark}   [2,100]{.mark}

  400                         [100,036]{.mark}   [2,105]{.mark}

  450                         [100,032]{.mark}   [2,106]{.mark}

  500                         [100,036]{.mark}   [2,108]{.mark}

  550                         [100,036]{.mark}   [2,107]{.mark}

  600                         [100,037]{.mark}   [2,109]{.mark}

  650                         [100,037]{.mark}   [2,109]{.mark}

  700                         [100,036]{.mark}   [2,109]{.mark}

  750                         [100,037]{.mark}   [2,110]{.mark}

  800                         [100,037]{.mark}   [2,110]{.mark}
  ---------------------------------------------------------------

# 4.3 RL & TOC-Based Simulation Model \[REVISED DRAFT -- pending author review\]

The TOC & RL-based simulation model was also built upon the virtual environment established in Section 4.1. The state space 𝒮 comprises 22 variables: two constraint-buffer variables --- the portion of constraint-buffer time not yet arrived at the CCR, H\^CB_wip,t, and the portion already queued at the CCR, H\^CB_queue,t --- together with, for each product i, its work-in-process level, W_i,t, and its finished-goods (shipping-buffer) level, H\^SB_i,t:

*𝒮 = { H\^CB_wip,t , H\^CB_queue,t } ∪ { W_i,t , H\^SB_i,t \| i = 1, ..., I }, I = 10*

H\^CB_wip,t and H\^CB_queue,t are measured in hours of processing time still owed to the CCR and take non-negative real values; W_i,t and H\^SB_i,t are non-negative integers. Before reaching the policy network, all state variables are normalized online using running mean and variance statistics accumulated during training.

This state informs the agent, which independently selects, for each product i, the number of units to release during the replenishment interval, forming the action space:

*𝒜(𝒮) = { M_i,t \| m = 0, 1, ..., 60 }, i = 1, ..., I = 10*

This decision is made autonomously and is not explicitly modeled by Equation 17. Because the I = 10 products are chosen independently and simultaneously, 𝒜(𝒮) is a multi-discrete space with 10 sub-actions of 61 categories each --- 610 discrete outputs in the policy's action layer. The upper bound M_i,t = 60 units was set based on the highest order size observed in the TOC-based model over \[xx\] replications \[confirm the scaling factor applied\]. The production sequence follows Equation 15, and, as in the TOC-based model, it is assumed that (ρ_i − c_i) = 1 when defining τ_i in Equation 10, since the original work by Atwater and Chakravorty (2002) does not take into account the revenue and variable costs of the products sold. The release of orders follows Equation 19. These relationships are illustrated in Figure 5.

The agent receives a numerical reward R_t+1 as a result of its action (Sutton and Barto, 2018), composed of a fixed baseline, a throughput term, and two penalty subfunctions:

*(20) R_t+1 = R_0 + TH_t+1 − P\^LS_t+1 − P\^W_t+1 − P\^SB_t+1*

R_0 is a fixed baseline reward added at every step. TH_t+1 is the throughput term; because the original work by Atwater and Chakravorty (2002) does not provide the information needed to compute throughput as sold value, TH_t+1 is set to zero during training, and its role is absorbed entirely into the lost-sales penalty P\^LS_t+1. Whenever demand for product i exceeds its current finished-goods level H\^SB_i,t, the system immediately fails to make a sale and cannot convert inventory into throughput (cf., Reference):

*(21) P\^LS_t+1 = λ_LS · Σ(i=1..I) L_i,t*

where L_i,t is the number of units of product i lost to unmet demand during interval t, and λ_LS is a penalty weight (λ_LS = 10 during training).

The second measure is inventory, represented by the total quantity of items rather than monetary values, for the same reason as throughput. Inventory is penalized through two subfunctions, one for work-in-process and one for finished goods, each summed across the I = 10 products:

*(22) P\^W_t+1 = λ_W · Σ(i=1..I) W_i,t*

*(23) P\^SB_t+1 = λ_SB · Σ(i=1..I) H\^SB_i,t*

with λ_W = 0.002 and λ_SB = 0.001. The constraint-buffer variables (H\^CB_wip,t, H\^CB_queue,t) are part of the state 𝒮 but are not penalized directly in the reward. The overall goal is to prevent lost sales while maintaining the lowest possible inventory level (cf., Schragenheim, 2010b); accordingly, λ_SB is set lower than λ_W.

In the RL & TOC-based simulation model, each decision step advances the simulation clock by the replenishment interval τ = 168 hours, matching the order-planning interval defined for the TOC-based model in Section 4.2. An episode begins at the start of a simulation run and proceeds through successive decision steps until either (i) the reward R_t+1 computed for a step turns negative, ending the episode immediately (cf. Reference), or (ii) the simulation clock reaches the 200,000-hour horizon, truncating the episode. By incorporating the reward function (Equation 20) into the return function (Equation 2) and assuming a discount factor γ = 0.99, along with rewards received at future time steps (t′ \> t), we construct the state-value function (Equation 3), from which the agent is trained to seek the optimal policy π\*.

The algorithm used to train the optimal policy is PPO (Schulman et al., 2017), implemented with the Stable-Baselines3 library (Raffin et al., 2021), using a multi-input policy network with two hidden layers of 128 units each, shared between the actor and critic branches, that approximates both the policy and the state-value function (Russell and Norvig, 2010; Schulman et al., 2017). Observations and rewards were normalized online (zero mean, unit variance, clipped at ±10,000) throughout training. Training used 8 parallel environment instances, a rollout length of 4,096 steps per environment, mini-batches of 2,048 transitions, and 5 optimization epochs per rollout, with a learning rate that decayed linearly from 10⁻³ to 10⁻⁴ (Equation 7) over a maximum of 16,800,000 timesteps and 277,777 episodes, the virtual environment being reset each time the step reward turned negative (cf. Reference). The maximum number of episodes was set based on \[define the reason\] (cf., Reference). Two criteria were used to terminate the training process (Reference): (i) the maximum number of timesteps was reached, or (ii) return saturation occurred, where the return remained stable over \[xx\] episodes.

From a computational perspective, the training process was completed in 2 hours, utilizing 15 machines running in parallel, each with the following specifications: an i5-13450HX processor, 16 GB of RAM, and an NVIDIA GeForce RTX 4050 graphics card. A state-value of 0.996 was achieved after 13,888 episodes, as shown in Figure 6a. The behavior of the penalty subfunctions is also illustrated in Figure 6b--d.

*\[Reviewer note: Figure 6's original caption lists four penalty-subfunction panels --- lost sales, work-in-process, CB level, and SB level ((b)--(e)) --- but the reward above has only three: lost sales, work-in-process, and SB/finished-goods (Equations 21--23), with no CB term. Panel (d), the CB-level curve, should be removed, and the current panel (e) (SB) should become the new panel (d), so the caption reads: (a) return; (b) lost-sales penalty; (c) work-in-process penalty; (d) finished-goods (SB) penalty. With this change, the body text's existing "Figure 6b--d" citation is already correct and needs no further edit.\]*

# ───── ORIGINAL TEXT BELOW (unchanged, kept for comparison) ─────

# 4.3 RL & TOC-Based Simulation Model

The TOC & RL-based simulation model was also built upon the virtual environment established in Section 4.1. Fifty variables were used to represent the state space of the environment at each time step $t$, i.e., $\mathcal{S =}\left\{ H_{i_{t}}^{OB},H_{i_{t}}^{CB},H_{i_{t}}^{SB},W_{i_{t}}^{CB},W_{i_{t}}^{SB}\ |\ h \in \mathbb{Z,\ }w = 0,\ 1,\ \ldots,\ 13 \right\}_{i = 1}^{I = 10}$. The true buffer levels $(h)$ can take any integer value, being negative when demand exceeds the buffer reference level, particularly for SB. The true values of work-in-process $(w)$ can range from 0 to 13, with 13 occurring when all resources $j$ in the longest production route are occupied (see [[Figure 2](#_Ref187516873)]{.mark} and Table S1). This state space informs the agent, which selects the number of products to produce during the replenishment interval from a set of [210]{.mark} possible actions, i.e., $\mathcal{A(S) =}\left\{ M_{i_{t}}\ |\ m = 0,\ 1,\ldots,\ 20 \right\}_{i = 1}^{I = 10}$. This decision is made autonomously and is not explicitly modeled by Equation 17. The quantity of each product to be produced ranges from 0 to [20]{.mark}, with 0 representing the decision to produce nothing. The maximum limit of [20]{.mark} was set based on the highest order size observed in the TOC-based model after [xx]{.mark} replications and then duplicated. The production sequence follows Equation 15, but similarly to the TOC-based model, it is assumed that $\left( \rho_{i} - c_{i} \right) = 1$ when defining $\tau_{i}$ in Equation 10, since the original work by Atwater and Chakravorty ([2002]{.mark}) does not take into account the revenue and variable costs of the products sold. The release of orders follows Equation 19. These relationships are illustrated in [[Figure 5](#_Ref187516938)]{.mark}.

![Diagrama Descrição gerada automaticamente](overleaf/figures/media/image5.png){width="6.295138888888889in" height="4.6030161854768155in"}

[]{#_Ref187516938 .anchor}[Figure 5. Outline of the RL & TOC-based simulation model.]{.mark}

The agent receives a numerical reward $R_{t + 1}$ as a result of its action (Sutton and Barto, 2018). Two of the performance measures from the TOC are implicitly used as penalty subfunctions within the following reward function:

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------
   $$R_{t + 1}\  = \frac{1}{I}\sum_{\forall i}^{}{1 - P_{i_{t + 1}}^{\tau} - P_{i_{t + 1}}^{W^{SB}} - P_{i_{t + 1}}^{H^{CB}} - P_{i_{t + 1}}^{H^{SB}}}.$$  \(20\)
  -------------------------------------------------------------------------------------------------------------------------------------------------------- ---------

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------

The first of these measures is throughput $(\tau)$, which, due to the original work by Atwater and Chakravorty ([2002]{.mark}) lacking information for its calculation, is treated as a penalty for lost sales ([cf. Reference]{.mark}). In other words, whenever the demand for a product exceeds its current level in the SB, the system immediately fails to make a sale and, consequently, cannot convert inventory into throughput ([cf., Reference]{.mark}). This penalty is incurred whenever the SB level is negative. Equation 21 outlines this condition, weighted by a factor of 0.3, which gives greater importance to this penalty:

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   $$P_{i_{t + 1}}^{\tau} = \left\{ \binom{0.3H_{i_{t}}^{SB}\ }{0\ \ \ \ \ \ \ \ \ \ \ \ }\binom{if\ H_{i_{t}}^{SB} < 0}{if\ H_{i_{t}}^{SB} \geq 0} \right.\ .$$  \(21\)
  --------------------------------------------------------------------------------------------------------------------------------------------------------------- ---------

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The second measure is inventory $\mathcal{(i)}$, which in the model is represented by the quantity of items rather than monetary values, for the same reason as throughput. Inventory is divided into three penalty subfunctions, each squared, to amplify the impact of larger values ([cf., Reference]{.mark}). The first subfunction penalizes work-in-process throughout the system, $P_{i_{t + 1}}^{W^{SB}}$. The second penalizes the level of the CB, $P_{i_{t + 1}}^{H^{CB}}$, while the third penalizes the number of items in the SB, $P_{i_{t + 1}}^{H^{SB}}$. The overall goal is to prevent lost sales while maintaining the lowest possible inventory level ([cf., Schragenheim, 2010b]{.mark}). For this reason, the weight is smaller in Equation 24, while the other equations have equal weights.

  -------------------------------------------------------------------------------
   $$P_{i_{t + 1}}^{W^{SB}} = \left( 0.02W_{i_{t}}^{SB} \right)^{2};$$  \(22\)
  --------------------------------------------------------------------- ---------
   $$P_{i_{t + 1}}^{H^{CB}} = \left( 0.02H_{i_{t}}^{CB} \right)^{2};$$  \(23\)

   $$P_{i_{t + 1}}^{H^{SB}} = \left( 0.01H_{i_{t}}^{SB} \right)^{2};$$  \(24\)
  -------------------------------------------------------------------------------

In the RL & TOC-based simulation model, the natural event that determines the length of each simulation run, or episode in reinforcement learning terminology ([cf., Reference]{.mark}), is the [72]{.mark}-hour replenishment interval $(\mathrm{\Delta}\varphi)$. By incorporating the reward function (Equation 21) into the return function (Equation 2) and assuming a discount factor $\gamma = 0.0$, along with rewards received at future time steps ($k = 71$), we can construct the state value function (Equation 3), from which the agent can be trained to seek the optimal policy, $\pi_{*}$.

The algorithm used for training the optimal policy is PPO ([Schulman et al., 2017]{.mark}), implemented using the Stable Baselines 3 library ([Raffin et al., 2021]{.mark}), which employs neural networks to approximate both the policy and the state-value function ([Russell and Norvig, 2010; Schulman et al., 2017]{.mark}). A learning rate of $\epsilon = 0.0002\$ (Equation 7) and a maximum number of 277,777 replications (or episodes) were used during training, with the virtual environment being reset each time the return reached zero ([cf. Reference]{.mark}). The maximum number of episodes was set based on \[[define the reason]{.mark}\] ([cf., Reference]{.mark}). Two criteria were used to terminate the training process ([Reference]{.mark}): (i) the maximum number of episodes was reached, or (ii) return saturation occurred, where the return remained stable over [xx]{.mark} episodes.

From a computational perspective, the training process was completed in 2 hours, utilizing 15 machines running in parallel, each with the following specifications: an i5-13450HX processor, 16 GB of RAM, and an NVIDIA GeForce RTX 4050 graphics card. A [state-value of 0.996]{.mark} was achieved after 13,888 episodes, as shown in [[Figure 6](#_Ref187516969)]{.mark}a. The behavior of penalty subfunctions is also illustrated in [[Figure 6](#_Ref187516969)]{.mark}b-d.

+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ![Gráfico Descrição gerada automaticamente](overleaf/figures/media/image6.png){width="4.521181102362204in" height="3.477214566929134in"}                                                              |
+:=================================================================================================:+:=================================================================================================:+
| [(a)]{.mark}                                                                                                                                                                                          |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| ![](overleaf/figures/media/image7.png){width="2.0263090551181104in" height="2.35in"}              | ![](overleaf/figures/media/image7.png){width="2.200112642169729in" height="2.349286964129484in"}  |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| [(b)]{.mark}                                                                                      | [(c)]{.mark}                                                                                      |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| ![](overleaf/figures/media/image7.png){width="2.0483180227471567in" height="2.348611111111111in"} | ![](overleaf/figures/media/image7.png){width="2.0483180227471567in" height="2.348611111111111in"} |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
| [(d)]{.mark}                                                                                      | [(e)]{.mark}                                                                                      |
+---------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+

[]{#_Ref187516969 .anchor}[Figure 6. Dynamic behaviour of: (a) the return function; (b) the penalty subfunction for lost sales; (c) the penalty subfunction for work-in-process; (d) the penalty subfunction for the CB level; and (e) the penalty subfunction for the SB level.]{.mark}

# 5. Execution of Simulation Experiments

The two simulation models --- one based on TOC and the other integrating RL with TOC --- were run across 52 replications (or episodes), representing one year of operation. Each replication simulated a 72-hour order planning interval, corresponding to a five-day workweek with two 7.2-hour shifts per day. The performance measures used as model outputs included: (i) the number of products produced and sold; (ii) the average number of parts in process; (iii) the average level of the CB; and (iv) the average level of the SB, all measured over a predefined replenishment interval. These metrics were captured both per product and in aggregate, generating a database from the simulation runs. Further details can be found in Tables S4-S11 of the supplementary material.

# 6 Summative and Formative Evaluations

The results were analyzed in two stages: (i) productivity analysis and (ii) behavior analysis. The productivity analysis compares the production performance of the two scheduling approaches, while the behavior analysis explains how these results were achieved.

To compare the performance of the two scheduling methods, hypothesis tests were conducted to determine whether there were statistically significant differences ([[Figure 7](#_Ref187517011)]{.mark}). Specifically, each test examined the null hypothesis that both scheduling methods come from the same probability distribution. Because at least one sample in each comparison failed the Shapiro--Wilk (normality) or Levene (homoscedasticity) tests, the Wilcoxon test was used instead of the two-sample t-test (Law, 2015; Telles *et al.*, 2020b). The Wilcoxon test was performed at a 95% confidence level and a 5% significance level ([]{.mark}[Table 3](#_Ref187517192)).

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   ![](overleaf/figures/media/image8.png){width="2.623161636045494in" height="1.968503937007874in"}  ![](overleaf/figures/media/image8.png){width="2.623161636045494in" height="1.968503937007874in"}
  -------------------------------------------------------------------------------------------------- --------------------------------------------------------------------------------------------------
                                             [(a)]{.mark}                                            [(b)]{.mark}

   ![](overleaf/figures/media/image8.png){width="2.623161636045494in" height="1.968503937007874in"}  ![](overleaf/figures/media/image8.png){width="2.623161636045494in" height="1.968503937007874in"}

                                             [(c)]{.mark}                                            [(d)]{.mark}
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[]{#_Ref187517011 .anchor}[Figure 7. Confidence intervals of the differences for: (a) the number of products produced and sold; (b) the average number of parts in process; (c) the average level of the CB; and (d) the average level of the SB.]{.mark}

The results indicate no statistically significant differences in demand (DM) or quantity of products sold (SP), showing that both approaches met demand equally. However, throughput time (FT), work in process (WIP), finished goods (FG), and total inventory (IV) levels are lower under the RL approach.

Behavioral analysis was conducted using three techniques to examine the relationship between state variables and the scheduling decisions. The first technique, Spearman's correlation, evaluates the monotonic relationship between two quantitative variables without requiring linearity (Barbosa and Azevedo, 2018; Xiao *et al.*, 2015). The correlation analysis shows that DBR exhibits a negative relationship between FG and WIP for the same product, reflecting the DBR mechanism that maintains the shipping buffer (SB) at its reference value regardless of production stage, thus balancing FG and WIP. The FG and WIP levels of different products behave similarly, except for products 7 and 9, which are unrestricted and released based on consumption. Actions showed no significant correlations with state variables, apart from weak correlations in products 2, 6, 7 (with FG), and product 9 (with WIP). The RL agent reveals a similar pattern regarding the relationship between FG and WIP, particularly for products 7 and 9 though the correlation is weaker. This implies that while the RL agent also seeks to balance FG and WIP, additional factors influence its decisions. Unlike DBR, the RL agent exhibits weak to moderate negative correlations between the action and the respective WIP (except for product 9). This negative correlation suggests that the RL agent prioritizes releasing materials when WIP is low, a behavior not observed in DBR. The results from both correlations can be found in appendix.

The second approach used linear regression to identify the contribution of WIP and FG on the quantity of products released by the scheduling methods. A linear regression was run for each product, treating WIP and FG as independent variables and the quantity of products released as the dependent variable (full table in appendix). The result shows that the DBR actions display a strong linear relationship with state variables, with coefficients of determination exceeding 80%. The primary influencing factors are the product's own WIP and FG levels, while the constraint buffer (CB) has no discernible impact. RL shows lower coefficients of determination, below 55%. Although WIP and FG remain the key decision variables, the influence of CB is again negligible. However, RL's decisions are less strongly determined by WIP and FG than DBR's, indicating additional complexities in the agent's decision-making process.

To complement the analysis and make RL decisions readable, a Decision Tree model was trained for each action based on the RL agent's state--action history. In DBR, WIP and FG collectively account for more than 85% of the decision's explanatory power. In the RL agent, WIP exerts more influence than FG, but combined they explain less than 50% of the decision-making process for each action ([]{.mark}[Figure 8](#_Ref187517031)). Unlike DBR, where other products' variables have minimal effect, several RL actions show up to 5% influence from variables related to other products, underscoring a more nuanced decision process.

![](overleaf/figures/media/image9.png){width="6.300694444444445in" height="2.9in"}

[]{#_Ref187517031 .anchor}Figure 8. Decision tree influence of variables on RL actions

The number of nodes in a decision tree reflects how many rules are required to explain the dataset (Myles *et al.*, 2004; Rudin *et al.*, 2022). In this study, the DBR decision trees use fewer nodes than their RL counterparts ([]{.mark}[Figure 9](#_Ref187517024)) indicating that the RL scheduling approach makes more complex decisions. For DBR, products 7 and 9 exhibit the simplest trees because they are not processed by the constraint, so their material release depends only on the respective shipping buffer (SB). Conversely, products 8 and 10 have the most complex trees, consistent with the fact that they pass through the constraint and involve more processing steps. In contrast, the RL decision trees do not show a clear pattern relating the number of nodes to the model's dynamics. Notably, product 9 has the largest tree in RL's case, despite not being processed by the constraint.

![](overleaf/figures/media/image10.png){width="6.300694444444445in" height="2.7604166666666665in"}

[]{#_Ref187517024 .anchor}Figure 9. Quantity of decision tree nodes.

# 7 Discussion

Integrating RL with DBR reduced inventory and shortened throughput times, thereby increasing production system efficiency by treating the shipping buffer (SB) as both WIP and FG. From a TOC standpoint, lower inventory boosts return on investment (ROI) and strengthens cash flow (Gupta *et al.*, 2002; Machado *et al.*, 2023). It also mitigates system variability by reducing queues at the resources (Goldratt, 2006a; Machado *et al.*, 2023). Shorter queues, in turn, improve lead times, allowing for quicker responses to demand fluctuations and machine breakdowns (Thürer *et al.*, 2022). By decomposing the SB into WIP and FG, this approach eliminates the need to maintain a fixed SB reference value and continuously adjust it, ultimately streamlining decision-making and enhancing production efficiency.

The principles of TOC were instrumental in shaping the RL model, particularly in defining the state and reward function. Since the state must provide all the information for the agent to make decisions (Sutton and Barto, 2018), incorporating the SB concept enables the RL agent to effectively understand the system\'s production needs and meet demand. This approach eliminates the need for comprehensive monitoring of the production system, such as tracking queued products (Shiue *et al.*, 2020; Thomas *et al.*, 2018), order characteristics (Liu *et al.*, 2022; Marchesano *et al.*, 2022), or the status and utilization of resources (Liu *et al.*, 2022; Sakr *et al.*, 2023). Furthermore, the reward function, designed based on TOC performance metrics (gain and inventory), has proven sufficient for training the RL algorithm and has demonstrated its effectiveness in decision-making.

Typically, RL scheduling methods are limited to scheduling predefined production orders (Esteso *et al.*, 2023). However, integrating RL with DBR allows the agent to determine order sizes while scheduling production based on the volume of finished goods in stock. This approach adapts RL to a make-to-stock (MTS) environment, contrasting with the more common focus on make-to-order (MTO) systems (Panzer and Bender, 2022). It also increases the autonomy of the scheduling process, aligning with the aim of autonomous decision-making in Industry 4.0 (Jan *et al.*, 2023; Oztemel and Gursev, 2020). Although DBR scheduling performed effectively in the simulation model, its integration with RL achieved equivalent productivity while reducing both inventory and throughput time ([]{.mark}[Table 3](#_Ref187517192)).

Correlation analysis, linear regression, and decision trees were used to explain the RL agent's decisions, with DBR serving as a baseline for comparison. These methods confirmed that DBR's behavior aligns with its underlying model, highlighting how SB levels influence product quantity decisions (Schragenheim, 2010c). In contrast, the RL agent prioritizes WIP more than FG when deciding what quantities to produce, showing a result of RL's capacity to learn the system's dynamics and adapt to disruptions such as machine breakdowns. This phenomenon is illustrated in [Figure 10](#_Ref187517040). When resource III experiences a breakdown (red bar at time T2), parts accumulate in front of the resource while the market continues to consume the FG. As FG levels decrease, a new order is released to restore the SB level, leading to an increase in WIP (time T3). By times T4 and T5, the levels of WIP and FG return to their initial values due to the protective capacity of the resources, and product releases continue in accordance with the SB level.

![Diagrama Descrição gerada automaticamente com confiança baixa](overleaf/figures/media/image11.png){width="4.689180883639545in" height="5.232854330708661in"}

[]{#_Ref187517040 .anchor}Figure 10. Machine breakdown in the DBR.

By highlighting RL\'s performance, it becomes evident that there is potential for improving DBR, particularly by considering WIP as a standalone decision-making variable rather than treating it solely as part of the SB. Although no direct relationship between WIP and the number of products released was identified, the decision tree's node count reveals a significant difference between the two scheduling methods. This discrepancy suggests that other variables or combinations of variables may influence decisions in certain system states. However, defining these relationships comprehensively can be both labor-intensive and, in some cases, infeasible (Kuhnle *et al.*, 2022; Rudin *et al.*, 2022).

# 8 Conclusions

This study represents a first attempt to integrate RL with DBR for production scheduling. The RL model was developed using TOC concepts and trained within a DES environment modeled on DBR studies, simulating a job shop with stochastic processes. Modifications, including machine setups and equipment breakdowns, were introduced to evaluate the solution's performance in scenarios with high variability. The results showed that RL achieved the same level of productivity as DBR but with lower inventory levels.

The primary contribution of this research lies in enhancing production scheduling efficiency by integrating RL with TOC principles and decomposing the SB into WIP and FG. Additionally, the analysis of RL behavior identified WIP as the key variable influencing product release decisions, highlighting opportunities to refine DBR by incorporating WIP more prominently. While the exact influence of WIP on RL actions could not be fully determined, integrating RL with DBR significantly improved the production system by reducing inventory and shortening throughput times. Furthermore, incorporating TOC concepts into RL modeling enabled effective decision-making without the need for exhaustive production monitoring.

Regarding limitations, this study's simulation model is theoretical, incorporating simplified representations of reality that exclude operational costs and system gains (Goldratt, 2006a). Consequently, economic analyses, including the TOC-proposed prioritization of product mix, were not performed. Additionally, the MTS operating mode used in this study did not account for the potential to operate in an MTO framework. Future research opportunities include implementing the proposed solution in more complex environments using empirical data, comparing the RL model with alternative approaches, and further exploring DBR by isolating WIP and FG as individual decision-making variables.

This study adopted a mechanism-based perspective to explore whether and how TOC can enhance Lean production theory in the context of optimizing a welding line in a manufacturing company. We applied Lean and TOC principles to design and implement an intervention that increases the economic efficiency of this line by more than 230 percent. The intervention challenged three assumptions underlying Lean implementation and sustainment: (i) inventory and capacity buffers are cost equivalent; (ii) reducing variability is the primary way to synchronize production with demand; and (iii) continuous improvement inherently leads to efficiency improvements. It also challenges the TOC assumption that increased demand has no limit, but reduced inventory and operating expenses do. As such, our study sheds light on the rationale for economic efficiency-centered design that goes beyond the Lean and TOC lenses. This rationale is grounded in the following kinematic mechanisms: (i) measuring production systems in terms of economic efficiency; (ii) synchronizing production and demand by eliminating obvious waste and reducing and/or buffering variability; and (iii) managing progress toward the economic efficiency frontier by means of system constraints.

# References

Atwater, J.B. and Chakravorty, S.S. (2002a), "A study of the utilization of capacity constrained resources in drum-buffer-rope systems", *Production and Operations Management*, Vol. 11 No. 2, pp. 259--273, doi: 10.1111/j.1937-5956.2002.tb00495.x.

Atwater, J.B. and Chakravorty, S.S. (2002b), "A study of the utilization of capacity constrained resources in drum-buffer-rope systems", *Production and Operations Management*, Vol. 11 No. 2, pp. 259--273, doi: 10.1111/j.1937-5956.2002.tb00495.x.

Barbosa, C. and Azevedo, A. (2018), "Assessing the impact of performance determinants in complex MTO / ETO supply chains through an extended hybrid modelling approach", Vol. 7543, doi: 10.1080/00207543.2018.1543970.

Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J. and Zaremba, W. (2016), "OpenAI Gym", pp. 1--4.

Buestán Benavides, M. and Van Landeghem, H. (2015), "Implementation of S-DBR in four manufacturing SMEs: a research case study", *Production Planning & Control*, Vol. 26 No. 13, pp. 1110--1127, doi: 10.1080/09537287.2015.1015060.

Cox III, J.F. and Schleier Jr., J.G. (2010), *Theory of Constraints Handbook*, edited by III Cox, J.F. and Jr. Schleier, J.G., Illustrate., McGraw-Hill.

Cox, J.F. and Schleier, J.G. (2010), *Theory of Constraints Handbook*, McGraw-Hill, New York, doi: 10.1002/cbdv.200490137.

Darlington, J., Francis, M., Found, P. and Thomas, A. (2015), "Design and implementation of a Drum-Buffer-Rope pull-system", *Production Planning and Control*, Vol. 26 No. 6, pp. 489--504, doi: 10.1080/09537287.2014.926409.

Eriksson, K., Ramasamy, S., Zhang, X., Wang, Z. and Danielsson, F. (2022), "Conceptual framework of scheduling applying discrete event simulation as an environment for deep reinforcement learning", *Procedia CIRP*, Vol. 107, pp. 955--960, doi: 10.1016/j.procir.2022.05.091.

Esteso, A., Peidro, D., Mula, J. and Díaz-Madroñero, M. (2023), "Reinforcement learning applied to production planning and control", *International Journal of Production Research*, Taylor & Francis, Vol. 61 No. 16, pp. 5772--5789, doi: 10.1080/00207543.2022.2104180.

Goldratt, E.M. (2006a), *The Haystack Syndrome: Sifting Information Out of the Data Ocean*, North River Press.

Goldratt, E.M. (2006b), *The Haystack Syndrome: Sifting Information Out of the Data Ocean*, North River Press.

Goldratt, E.M. and Cox, J. (1984a), *The Goal: Excellence In Manufacturing*, North River Press, Croton-on-Hudson, N.Y.

Goldratt, E.M. and Cox, J. (1984b), *The Goal: Excellence In Manufacturing*, North River Press, Croton-on-Hudson, N.Y.

Golmohammadi, D. (2015), "A study of scheduling under the theory of constraints", *International Journal of Production Economics*, Management Science and Information Systems Department, University of Massachusetts Boston, United States, Vol. 165, pp. 38--50, doi: 10.1016/j.ijpe.2015.03.015.

Guo, P., Shi, H., Wang, Y. and Xiong, J. (2024), "Multi-objective scheduling of cloud-edge cooperation in distributed manufacturing via multi-agent deep reinforcement learning", *International Journal of Production Research*, doi: 10.1080/00207543.2024.2329316.

Gupta, M., Ko, H.-J. and Min, H. (2002), "TOC-based performance measures and five focusing steps in a job-shop manufacturing environment", *International Journal of Production Research*, Department of Management, School of Business, University of Louisville, Louisville, KY 40292, United States, Vol. 40 No. 4, pp. 907--930, doi: 10.1080/00207540110097185.

Gupta, M.C. and Boyd, L.H. (2008), "Theory of constraints: A theory for operations management", *International Journal of Operations and Production Management*, Vol. 28 No. 10, pp. 991--1012, doi: 10.1108/01443570810903122.

Hopp, W.J. and Spearman, M.S. (2021), "The lenses of lean: Visioning the science and practice of efficiency", *Journal of Operations Management*, Vol. 67 No. 5, pp. 610--626, doi: 10.1002/joom.1115.

Jan, Z., Ahamed, F., Mayer, W., Patel, N., Grossmann, G., Stumptner, M. and Kuusk, A. (2023), "Artificial intelligence for industry 4.0: Systematic review of applications, challenges, and opportunities", *Expert Systems with Applications*, Elsevier Ltd, Vol. 216 No. November 2021, p. 119456, doi: 10.1016/j.eswa.2022.119456.

Kuhnle, A., May, M.C., Schäfer, L. and Lanza, G. (2022), "Explainable reinforcement learning in production control of job shop manufacturing system", *International Journal of Production Research*, Taylor & Francis, Vol. 60 No. 19, pp. 5812--5834, doi: 10.1080/00207543.2021.1972179.

Lang, S., Kuetgens, M., Reichardt, P. and Reggelin, T. (2021), "Modeling Production Scheduling Problems as Reinforcement Learning Environments based on Discrete-Event Simulation and OpenAI Gym", *IFAC-PapersOnLine*, Vol. 54 No. 1, pp. 793--798, doi: 10.1016/j.ifacol.2021.08.093.

Law, A.M. (2015), *Simulation Modeling and Analysis*, 5th ed., McGraw-Hill, Tucson.

Liu, R., Piplani, R. and Toro, C. (2022), "Deep reinforcement learning for dynamic scheduling of a flexible job shop", *International Journal of Production Research*, Vol. 60 No. 13, pp. 4049--4069, doi: 10.1080/00207543.2022.2058432.

Ma, L. and Chen, J. (2009), "Modeling and Simulating of Time Buffer Control for Engine Remanufacturing System", *2009 International Conference on Information Engineering and Computer Science*, IEEE, pp. 1--4, doi: 10.1109/ICIECS.2009.5365624.

Mabin, V.J. and Balderstone, S.J. (2003), "The performance of the theory of constraints methodology: Analysis and discussion of successful TOC applications", *International Journal of Operations and Production Management*, Vol. 23 No. 5--6, pp. 568--595, doi: 10.1108/01443570310476636.

Machado, M.P., Abreu, J.L., Morandi, M.I.M., Piran, F.S. and Lacerda, D.P. (2023), "Exploratory decision robustness analysis of the theory of constraints focusing process using system dynamics modeling", *International Journal of Production Economics*, Elsevier B.V., Vol. 260 No. March, p. 108856, doi: 10.1016/j.ijpe.2023.108856.

Marchesano, M.G., Guizzi, G., Popolo, V. and Converso, G. (2022), "Dynamic scheduling of a due date constrained flow shop with Deep Reinforcement Learning", *IFAC-PapersOnLine*, Vol. 55 No. 10, pp. 2932--2937, doi: 10.1016/j.ifacol.2022.10.177.

Márquez, C.R.H. and Ribeiro, C.C. (2022), "Shop scheduling in manufacturing environments: a review", *International Transactions in Operational Research*, Vol. 29 No. 6, pp. 3237--3293, doi: 10.1111/itor.13108.

Myles, A.J., Feudale, R.N., Liu, Y., Woody, N.A. and Brown, S.D. (2004), "An introduction to decision tree modeling", pp. 275--285, doi: 10.1002/cem.873.

Nguyen, T., Duong, Q.H., Nguyen, T. Van, Zhu, Y. and Zhou, L. (2022), "Knowledge mapping of digital twin and physical internet in Supply Chain Management: A systematic literature review", *International Journal of Production Economics*, Elsevier B.V., Vol. 244 No. July 2021, p. 108381, doi: 10.1016/j.ijpe.2021.108381.

Oztemel, E. and Gursev, S. (2020), "Literature review of Industry 4 . 0 and related technologies", *Journal of Intelligent Manufacturing*, Springer US, Vol. 31 No. 1, pp. 127--182, doi: 10.1007/s10845-018-1433-8.

Panzer, M. and Bender, B. (2022), "Deep reinforcement learning in production systems: a systematic literature review", *International Journal of Production Research*, Taylor & Francis, Vol. 60 No. 13, pp. 4316--4341, doi: 10.1080/00207543.2021.1973138.

Rudin, C., Chen, C., Chen, Z., Huang, H., Semenova, L. and Zhong, C. (2022), "Interpretable machine learning: Fundamental principles and 10 grand challenges", *Statistics Surveys*, Vol. 16, pp. 1--85, doi: 10.1214/21-SS133.

Sakr, A.H., Aboelhassan, A., Yacout, S. and Bassetto, S. (2023), "Simulation and deep reinforcement learning for adaptive dispatching in semiconductor manufacturing systems", *Journal of Intelligent Manufacturing*, Springer, Vol. 34 No. 3, pp. 1311--1324, doi: 10.1007/s10845-021-01851-7.

Schragenheim, A. (2010a), "Supply Chain Management", *Theory of Constraints Handbook*.

Schragenheim, E. (2010b), "From DBR to Simplified-DBR for Make-to-Order", in Cox III, J.F. and Schleier Jr., J.G. (Eds.), *Theory of Constraints Handbook*, McGraw-Hill.

Schragenheim, E. (2010c), "Managing Make-to-Stock and the Concept of Make-to-Availability", in III Cox, J.F. and Jr. Schleier, J.G. (Eds.), *Theory of Constraints Handbook*, McGraw-Hill.

Schragenheim, E. and Ronen, B. (1990), "Drum-Buffer-Rope Shop Floor Control", *Production and Inventory Management*, Vol. 31 No. 2, pp. 18--22.

Sellitto, M.A. and Brusius, W. (2017), "Maintenance strategy based on reliability analytical models for three parallel mechanical transformation machines", *IEEE Latin America Transactions*, Vol. 15 No. 5, doi: 10.1109/TLA.2017.7912597.

Shiue, Y.-R., Lee, K.-C. and Su, C.-T. (2020), "A Reinforcement Learning Approach to Dynamic Scheduling in a Product-Mix Flexibility Environment", *IEEE Access*, Vol. 8, pp. 106542--106553, doi: 10.1109/ACCESS.2020.3000781.

Stefano, G. da S., Antunes, T. dos S., Lacerda, D.P., Wolf Motta Morandi, M.I. and Piran, F.S. (2022), "The impacts of inventory in transfer pricing and net income: Differences between traditional accounting and throughput accounting", *British Accounting Review*, Elsevier Ltd, Vol. 54 No. 2, p. 101001, doi: 10.1016/j.bar.2021.101001.

Sutton, R.S. and Barto, A.G. (2018), *Reinforcement Learning: An Introduction*, Vol. 258, Bradford Books.

Telles, E.S., Lacerda, D.P., Morandi, M.I.W., Ellwanger, R., de Souza, F.B. and Piran, F.S. (2022a), "Drum-Buffer-Rope in an engineering-to-order productive system: a case study in a Brazilian aerospace company", *Journal of Manufacturing Technology Management*, Vol. 33 No. 6, pp. 1190--1209, doi: 10.1108/JMTM-10-2021-0420.

Telles, E.S., Lacerda, D.P., Morandi, M.I.W., Ellwanger, R., de Souza, F.B. and Piran, F.S. (2022b), "Drum-Buffer-Rope in an engineering-to-order productive system: a case study in a Brazilian aerospace company", *Journal of Manufacturing Technology Management*, Vol. 33 No. 6, pp. 1190--1209, doi: 10.1108/JMTM-10-2021-0420.

Telles, E.S., Lacerda, D.P., Morandi, M.I.W.M. and Piran, F.A.S. (2020a), "Drum-buffer-rope in an engineering-to-order system: An analysis of an aerospace manufacturer using data envelopment analysis (DEA)", *International Journal of Production Economics*, Vol. 222 No. February, p. 107500, doi: 10.1016/j.ijpe.2019.09.021.

Telles, E.S., Lacerda, D.P., Morandi, M.I.W.M. and Piran, F.A.S. (2020b), "Drum-buffer-rope in an engineering-to-order system: An analysis of an aerospace manufacturer using data envelopment analysis (DEA)", *International Journal of Production Economics*, Elsevier B.V., Vol. 222 No. February 2019, p. 107500, doi: 10.1016/j.ijpe.2019.09.021.

Thomas, T.E., Koo, J., Chaterji, S. and Bagchi, S. (2018), "Minerva: A reinforcement learning-based technique for optimal scheduling and bottleneck detection in distributed factory operations", *2018 10th International Conference on Communication Systems & Networks (COMSNETS)*, Vol. 2018-Janua, IEEE, pp. 129--136, doi: 10.1109/COMSNETS.2018.8328189.

Thürer, M., Fernandes, N.O. and Stevenson, M. (2022), "Production planning and control in multi-stage assembly systems: an assessment of Kanban, MRP, OPT (DBR) and DDMRP by simulation", *International Journal of Production Research*, Taylor & Francis, Vol. 60 No. 3, pp. 1036--1050, doi: 10.1080/00207543.2020.1849847.

Watson, K.J., Blackstone, J.H. and Gardiner, S.C. (2007), "The evolution of a management philosophy: The theory of constraints", *Journal of Operations Management*, Vol. 25 No. 2, pp. 387--402, doi: 10.1016/j.jom.2006.04.004.

Wu, H.H., Chen, C.P., Tsai, C.H. and Yang, C.J. (2010), "Simulation and scheduling implementation study of TFT-LCD Cell plants using Drum-Buffer-Rope system", *Expert Systems with Applications*, Elsevier Ltd, Vol. 37 No. 12, pp. 8127--8133, doi: 10.1016/j.eswa.2010.05.075.

Xiao, C., Ye, J., Esteves, R.M. and Rong, C. (2015), "Using Spearman ' s correlation coef fi cients for exploratory data analysis on big dataset", doi: 10.1002/cpe.

Yu, X. and Ram, B. (2006), "Bio-inspired scheduling for dynamic job shops with flexible routing and sequence-dependent setups", *International Journal of Production Research*, Vol. 44 No. 22, pp. 4793--4813, doi: 10.1080/00207540600621094.

 
