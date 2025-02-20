# NEIPR_Fractal_Ellipses
**Nested-Ellipse-Intersection-Point-Recursion**

![image](https://github.com/user-attachments/assets/4c2bcabf-3fc9-4433-9d3d-dc4c8023b9d4)


 
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

Digging deeper into the tools psychologists use to study personality, I stumbled upon Mahalanobis distance. I fell in love with it. Its ability to measure distances in multivariate space felt like a breakthrough, but it didn’t solve all my problems.  

Ultimately, we converged on a proprietary solution I can’t disclose due to an NDA. But deep down, I felt there was a broader mathematical challenge here—one that could apply to similar problems beyond our specific use case.  

A year later, while staring at Mahalanobis distance long enough to lose track of time, I noticed something: the ellipse rings it produced reminded me of the constrained chord-sum properties of ellipses.  

Ah ha!  

What if we could optimize weights without normalization by mapping them to points on an ellipse’s perimeter? The bi-foci chords of the ellipse could then serve as a constrained distribution for percentage matches.  

But there was a catch. An ellipse only has two chords, limiting us to two weights. Using multiple ellipses would force us to work with an even number of weights, and optimizing weights across different ellipses would create disjunctions. Plotting more points on a single ellipse faced the same issue: we’d still need an even number of weights.  

This led me to a critical question: How could we borrow just one of the chords to represent our weights?  

And that’s where I introduce you to Nested Ellipse Intersection Point Recursion.  

![image](https://github.com/user-attachments/assets/062ad4a9-af8c-4a2c-bf6b-7cf43ce5cc92)

![image](https://github.com/user-attachments/assets/65597836-eeba-4cdd-a5a2-137e2f65166a)


**Chapter 1: Introduction**

**1.1 Motivation: Exploring Ellipse Recursion**

This research investigates the properties and potential applications of a recursive process involving ellipses. The process generates a sequence of nested ellipses, where the foci of each ellipse serve as the centers of the next generation of ellipses. This recursive construction leads to intriguing geometric patterns and relationships that warrant further mathematical analysis.

![image](https://github.com/user-attachments/assets/6ae71730-a890-4490-a5f3-ec7f204d9234)

![image](https://github.com/user-attachments/assets/e79f9ac3-1aec-483d-8f3f-fe5f8a9a475f)

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


```python
# @title #Scaling to Dual Foci Nested Ellipse Recursion
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive_output, FloatSlider, Button, VBox, HBox, Tab, HTML, IntSlider
from IPython.display import display

def generate_ellipse(center, a, b, theta):
    """Generate (x, y) points for an ellipse."""
    x = center[0] + a * np.cos(theta)
    y = center[1] + b * np.sin(theta)
    return x, y

def calculate_foci(center, a, b):
    """Calculate the foci of an ellipse with center, semi-major axis a, and semi-minor axis b."""
    if a < b:
        raise ValueError("Semi-major axis (a) must be ≥ semi-minor axis (b).")
    c = np.sqrt(a**2 - b**2)
    return (center[0] - c, center[1]), (center[0] + c, center[1])

def plot_ellipses(a, b, scale_factor, recursion_depth):
    """Plot the main ellipse and recursively plot scaled ellipses at the foci."""
    try:
        if a < b:
            raise ValueError("Parameter 'a' must be ≥ parameter 'b'.")

        plt.figure(figsize=(10, 8))

        def recursive_plot(center, a, b, depth):
            if depth == 0:
                return

            theta = np.linspace(0, 2 * np.pi, 500)
            x, y = generate_ellipse(center, a, b, theta)
            plt.plot(x, y, label=f'Depth {recursion_depth - depth}')

            focus1, focus2 = calculate_foci(center, a, b)
            plt.scatter(*focus1, color='red', s=20)
            plt.scatter(*focus2, color='red', s=20)

            new_a = a * scale_factor
            new_b = b * scale_factor
            recursive_plot(focus1, new_a, new_b, depth - 1)
            recursive_plot(focus2, new_a, new_b, depth - 1)

        recursive_plot((0, 0), a, b, recursion_depth)

        plt.title(f'Recursive Ellipses (Scale={scale_factor:.2f}, Depth={recursion_depth})')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()

        c = np.sqrt(a**2 - b**2)
        auto_scale = 1 - c / a
        print(f"Auto-calculated Scale Factor (K = 1 - c/a): {auto_scale:.2f}")
    except Exception as ex:
        print("Error in plot_ellipses:", str(ex))

def interactive_plot():
    """Create interactive widgets for parameters and auto-scaling."""
    a_slider = FloatSlider(value=5.0, min=1.0, max=10.0, step=0.1, description='Semi-major (a):')
    b_slider = FloatSlider(value=3.0, min=1.0, max=10.0, step=0.1, description='Semi-minor (b):')
    scale_slider = FloatSlider(value=0.6, min=0.1, max=0.9, step=0.01, description='Scale Factor:')
    depth_slider = IntSlider(value=3, min=1, max=5, step=1, description='Recursion Depth:')
    autoscale_btn = Button(description='Auto Scale K')

    def update_scale(_):
        try:
            a_val = a_slider.value
            b_val = b_slider.value
            if a_val < b_val:
                print("Error: a must be ≥ b")
                return
            c_val = np.sqrt(a_val**2 - b_val**2)
            auto_K = 1 - c_val / a_val
            scale_slider.value = auto_K
        except Exception as ex:
            print("Error in autoscale:", str(ex))

    autoscale_btn.on_click(update_scale)

    ui = VBox([a_slider, b_slider, scale_slider, depth_slider, autoscale_btn])
    out = interactive_output(plot_ellipses, {'a': a_slider, 'b': b_slider, 'scale_factor': scale_slider, 'recursion_depth': depth_slider})
    return ui, out

def main():
    """Run the interactive plot."""
    try:
        ui, out = interactive_plot()
        display(ui, out)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
```

### Introduction to Dual Foci Nested Ellipse Visualization
This Python code creates an interactive visualization of recursively nested ellipses using Jupyter widgets and matplotlib. The program demonstrates geometric recursion by generating smaller ellipses at the foci of parent ellipses, with parameters controlled through sliders for semi-major/minor axes (a/b), scale factor, and recursion depth. Key features include automatic foci calculation, real-time parameter adjustments, and an "Auto Scale K" button that calculates an optimal scaling factor based on the ellipse's eccentricity. The visualization helps explore fractal-like patterns emerging from the dual-focus recursion principle, with color-coded depth levels and mathematical validation of geometric constraints. Built with numpy for calculations and ipywidgets for interactivity, this implementation serves as both an educational tool for conic section studies and a foundation for more complex recursive geometric systems.

---
Answer from Perplexity: pplx.ai/share

![image](https://github.com/user-attachments/assets/a66bcb0b-842b-4aac-9348-9c3877aa5b2b)

![image](https://github.com/user-attachments/assets/b166df8a-de7b-4488-9e34-305043f3dad5)

![image](https://github.com/user-attachments/assets/4fd9130b-45d0-4b7f-8bfa-b5d46f92c7e3)

![image](https://github.com/user-attachments/assets/57726125-0ac0-4d08-b5de-8fc17463c498)


