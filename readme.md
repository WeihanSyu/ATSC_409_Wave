# Particle Movement Through a Shallow Water Wave Field
**Model and code explanation provided in the pdf**

Consider a rectangular area with all sides enclosed:
* There are two points in the area that will periodically generate waves.
* These points can be considered source points and they both will generate a wave with the exact same amplitude at any given time.
* Next, consider a ball of some mass and initial velocity.
* This project will focus on the movement of the ball through the wave field using simple momentum equations to derive the interaction between the water and the ball.

<details>
<summary><b>Code snippet</b></summary>

```python
surf = ax.plot_surface(X, Y, h.field[i, 0:(m+1):2, 0:(n+1):2], alpha=0.4, cmap='Blues', linewidth=0, antialiased=False) 
surf = ax.plot_surface(x + xspace[i], y + yspace[i], z - (dx / 2), color='black')  
```
</details>

* The output of this script consists of two animations. One for the wave field and one for the ball. 
* Using them in conjunction makes the animation quite choppy.
* Turn off the ball to see a smooth wave field animation.
