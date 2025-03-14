# NEIPR_Fractal_Ellipses
**Nested-Ellipse-Intersection-Point-Recursion**
 ![4ebe0e6c-8b3c-4f7b-bdd2-b2a43cf42dc7](https://github.com/user-attachments/assets/8fc42076-85b7-4649-a447-b002a05be925)


Neat visual, right?  

I’m not a mathematician by trade, but I find immense joy in studying it. A few years ago, I found myself at a startup tasked with building their recommendation engine—a personality-driven matching algorithm designed for their app. The promise to users was clear: connect people based on deep compatibility, rooted in personality insights. But this promise set the stage for a fascinating and frustrating dilemma.  

At its core, the algorithm had to justify the degree of compatibility between any two users using personality results. These results, derived from established frameworks like OCEAN (Big Five) or Myers-Briggs, needed to be translated into explainable percentage breakdowns—e.g., "30% compatible in factor X, 20% in factor Y"—and then averaged into an overall match score.  

Transparency was non-negotiable. Each percentage had to be explainable to users, with no room for guesswork or misleading claims. This demand for rigor pushed us to think deeply about how we designed the system.  

But there were constraints—many of them.  

First, we couldn’t rely on a pure Stable Matching Algorithm. While stable matching works well for finite, one-to-one pairings, our system needed to provide a continuous flow of recommendations. Users had to remain open to new matches without ever being "locked in," meaning the algorithm could never fully converge on a stable state. The flow of recommendations had to behave such that neither side's availability for more matches would ever terminate them from being recommended to or receiving recommendations for more users.

Second, standard collaborative filtering (or hybrid models) was off the table. As a personality-first model, we were explicitly constrained against perpetuating popularity biases in the user pool. Recommendations couldn’t favor "popular" users over others, as this would undermine the integrity of personality-based matching.  

Third, reinforcement learning or bandit algorithms weren’t viable either. With an action space (or bandit arms) growing to the size of our user base, we’d face massive cold-start problems. The inherent turn-taking mechanism of exploration vs. exploitation simply wouldn’t scale.  

We explored clustering users by personality types as a starting point to mitigate these issues. While this helped, it didn’t solve everything. In practice, it opened up a web of increasingly complex directions, each more daunting than the last.  

From the outset, I knew some form of multi-objective optimization would be essential. But when I experimented with linear optimization methods, I ran into a major issue: explainability. Watching the weights shift with each gradient update was one thing, but seeing those gradients dampened by normalization—and then watching the weights mutate further—was another. The process distorted the clarity we needed to explain matches to users.  

But normalization seemed unavoidable, right? How else could we constrain compatibility scores to percentages (out of 100%) after optimization?  

This question lingered, and then inspiration struck from unexpected places. I thought about pulley systems, where tension is distributed across different segments of a single rope. Then I recalled the Dirichlet stick-breaking process, which can enforce a beta distribution—potentially onto my percentage matches.  

Digging deeper into the tools psychologists use to study personality, I stumbled upon **Mahalanobis distance**. I fell in love with it. Its ability to measure distances in multivariate space felt like a breakthrough, but it didn’t solve all my problems.  

![image](https://github.com/user-attachments/assets/37bf1e70-2a25-4226-999b-a3d0366ef43f)

Ultimately, we converged on a proprietary solution I can’t disclose due to an NDA. But deep down, I felt there was a broader mathematical challenge here—one that could apply to similar problems beyond our specific use case.  

A year later, while staring at Mahalanobis distance long enough to lose track of time, I noticed something: the ellipse rings it produced reminded me of the constrained chord-sum properties of ellipses.  

![image](https://github.com/user-attachments/assets/c64084ed-fc81-4a47-8ba2-acbe137f5eb0)

Ah ha!  

What if we could optimize weights without normalization by mapping them to points on an ellipse’s perimeter? The bi-foci chords of the ellipse could then serve as a constrained distribution for percentage matches.  

But there was a catch. An ellipse only has two chords, limiting us to two weights. Using multiple ellipses would force us to work with an even number of weights, and optimizing weights across different ellipses would create disjunctions. Plotting more points on a single ellipse faced the same issue: we’d still need an even number of weights.  

This led me to a critical question: How could we borrow just one of the chords to represent our weights?  

And that’s where I introduce you to Nested Ellipse Intersection Point Recursion.  

![image](https://github.com/user-attachments/assets/062ad4a9-af8c-4a2c-bf6b-7cf43ce5cc92)

---

**The Full Break Down**

![image](https://github.com/user-attachments/assets/4c2bcabf-3fc9-4433-9d3d-dc4c8023b9d4)

---

**Table of Contents**
.
.
.

**Chapter 1: Introduction**

**1.1 Motivation: Exploring Ellipse Recursion**

This research investigates the properties and potential applications of a recursive process involving ellipses. The process generates a sequence of nested ellipses, where the foci of each ellipse serve as the centers of the next generation of ellipses. This recursive construction leads to intriguing geometric patterns and relationships that warrant further mathematical analysis.

![image](https://github.com/user-attachments/assets/6ae71730-a890-4490-a5f3-ec7f204d9234)



**1.2 Key Findings**

The exploration of ellipse recursion has yielded several key findings:

* **Exponential Growth:** The number of ellipses, foci, and chords generated through the recursive process increases exponentially with each level of recursion.
* **Self-Similarity:** The nested ellipses exhibit self-similar structures, characteristic of fractal geometry.
* **Dynamic Behavior:** The chords and tangent lines associated with the ellipses demonstrate dynamic and coordinated movement as points on the ellipses' perimeters are varied.
* **Number Sequences:**  Analysis of the recursive process has revealed number sequences related to the number of ellipses, chords, and foci at each level. These sequences exhibit connections to established mathematical equations, such as the Ramanujan-Nagell equation and perfect number equations.

![image](https://github.com/user-attachments/assets/497aefeb-5779-4a22-b462-0904fad3656d)

**1.3 Potential Applications**

The geometric patterns and number sequences identified in ellipse recursion suggest potential applications in various domains:

* **Hierarchical Dirichlet Process:**  The recursive structure could offer insights for improving the Hierarchical Dirichlet Process, a Bayesian nonparametric model used in machine learning.
* **Decision Tree Node Splits:** The geometric properties of the recursive ellipses could be leveraged to develop new splitting criteria for decision trees, potentially enhancing their efficiency.
* **Quantum-Resistant Cryptography:** The complex patterns generated through ellipse recursion might have applications in designing encryption schemes resistant to quantum computing attacks.
* 
![image](https://github.com/user-attachments/assets/f883534d-60fe-444a-a58f-0aa7270a46f0)

**1.4 Research Objectives**

This research aims to:

* **Formalize Ellipse Recursion:**  Provide a rigorous mathematical framework for describing the ellipse recursion process, including the scaling factors and relationships between different geometric elements.
* **Analyze Number Patterns:**  Investigate the number sequences arising from ellipse recursion and establish their connections to known mathematical equations.
* **Explore Applications:**  Further explore the potential applications of ellipse recursion in machine learning, decision tree algorithms, and cryptography.

**1.5 Chapter Overview**

The subsequent chapters will delve into the mathematical details of ellipse recursion, exploring its properties, patterns, and potential applications. We will analyze the geometric relationships, number sequences, and computational aspects of this recursive process, and discuss its implications for various fields of mathematics and computer science.

![image](https://github.com/user-attachments/assets/a66bcb0b-842b-4aac-9348-9c3877aa5b2b)

![image](https://github.com/user-attachments/assets/b166df8a-de7b-4488-9e34-305043f3dad5)

![image](https://github.com/user-attachments/assets/4fd9130b-45d0-4b7f-8bfa-b5d46f92c7e3)

![image](https://github.com/user-attachments/assets/57726125-0ac0-4d08-b5de-8fc17463c498)


