# gpu_raytracing

This repository is an implementation of ray tracing with CUDA.

The source codes are mainly based (some parts, e.g., noise, are chatGPT support) on ray tracing one weekend series [link](https://raytracing.github.io/), but written in the CUDA with OpenGL (for visualization purposes).

If you have questions or suggestions, please feel free to comment on the GitHub issues.

## How to run?

```bash
cd src
make
```

Then,

```bash
main
```

Generated scenes.
![cornell box](figures/cornell_box.png)
![cornell smoke](figures/cornell_box_smoke.png)
