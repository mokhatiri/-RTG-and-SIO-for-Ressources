# Description:
this is a Random terrain generation and swarm intelligence optimisation project visualisation project using javaFX.

For now the implimentation is 2D;

## Visualization (JavaFX)

This project uses JavaFX to present interactive previews for terrain/noise generation and PSO visualization. The `app.Launcher` opens a simple interface to explore the 2D previews and tweak parameters in real time via the controls. Use the `NoisePreview2D` and `PSOPreview` views for quick visual exploration.

Run the application (Windows example):
```pwsh
mvn javafx:run -Djavafx.platform=win
```

# People:
@mokhatiri
@lamseey

# Project Steps:



## n-Dimensional (ND) generalization

To make the core components more flexible and future-proof, the project includes lightweight n-dimensional array wrappers and ND-capable variations of key algorithms. The 2D view and behaviour are kept backward-compatible.

- New utility types:
    - `DoubleNDArray` and `IntNDArray` — flat-backed ND containers with `from2D(...)`, `from3D(...)` factories and general `get`/`set`.
    - `PositionMapper` — converts continuous PSO positions into discrete ND coordinates.

- ND-enabled algorithmic changes:
    - `TerrainAnalyzer(DoubleNDArray)` supports `computeSlopeND()` and `categorizeTerrainND(...)` with neighborhood voting.
    - `NaturalResourceRandomizer` now exposes `randomizeResourceWeightedND(IntNDArray terrain, DoubleNDArray flatness, double[] baseProbabilities)` that returns an `IntNDArray` whose last axis contains resource presence flags.
    - Fitness functions (`ResourceFitness`, `TerrainFitness`) implement `evaluate(int[] coords)` to operate on ND coordinates.
    - `ResourcePlacementSolution` now stores `int[] coords` for ND placement.

Snippet examples:

```java
// build an ND heightmap from a 2D array
DoubleNDArray heightND = DoubleNDArray.from2D(heightMap);
TerrainAnalyzer analyzerND = new TerrainAnalyzer(heightND);
DoubleNDArray slopeND = analyzerND.computeSlopeND();
IntNDArray terrainND = analyzerND.categorizeTerrainND(waterLevel, hillLevel, mountainLevel, transition);

// generate ND resource placement
NaturalResourceRandomizer rr = new NaturalResourceRandomizer(seed, width, height, scale, offsetX, offsetY);
IntNDArray resourceND = rr.randomizeResourceWeightedND(terrainND, slopeND, params.getProbabilitiesArray());

// use swarm optimizer with ND-aware discrete fitness via an adapter
int[] gridShape = new int[]{width, height};
ResourceFitness rFitness = new ResourceFitness(flatness2D, flatnessCoeff, resourceTerrain3D, /*values...*/);
TerrainFitness tFitness = new TerrainFitness(flatness2D, flatnessCoeff, terrain2D, /*values...*/);
SwarmOptimizer optimizer = new SwarmOptimizer(dimensions, w, c1, c2, resourceCoeff, terrainCoeff, rFitness, tFitness, gridShape);
```

---
# 1. Terrain:

-the terrain is generated using noise map ( terrain/ Noise.java , NoiseMapGenerator.java )
this is done using OpenSimplex Noise (for now)


## Simplex noise:

the main difference between perlin, and simplex noise is that perlin uses interpolation, simplex does summation, that's why, simplex has a complexity of O(n^2), while perlin is O(2^n) complex. Also perlin uses a grid, but simplex uses simplexes, in 2d that's a triangle, 3d that's a tetrahedron, ..etc.

Simplex noise uses a direct summation from the corners's contributions of the simplex; where the contribution is a multiplication of the extrapolation of the gradient ramp and a radially symmetric attenuation function.

`- Contribution=(gradient ramp)×(attenuation)`

At each corner of the simplex:
    A random gradient vector is assigned.
    The algorithm computes how aligned the point is with that vector, using:
    ` ramp= gradient_corner ⋅(x_current − x_corner​) `

And the attenuation function is basically the "fade-out" factor:
we are using something like:
 - `t=(radius^2−distance^2)`
 - `attenuation = t^4`

so basically:
```
noise = sum[corners] (g.d).t^4
```

while simplex is not openSource, because it has a patent.
But:
OpenSimplex is very similar to Simplex noise in structure and math, but it was created as a patent-free alternative with lower directional artifacts.

### Difference:

1. Open Simplex uses a Different lattice, it uses a different skewing matrix that produces a slightly different simplex tiling of space. (but still based on simplex-like shapes)

2. 
```
 noise = sum[corners] attenuation(d).(gradient.d) 
```

### for 2D:

the skew matrices are:

2D skew : 
```
s=(x+y)⋅α
i=⌊x+s⌋, j=⌊y+s⌋
```

2D unskew:
```
t=(i+j)⋅β
X0​=i−t, Y0​=j−t
x0​=x−X0​, y0​=y−Y
```

where:
α = (1/√3 - 1)/2
β = (3 - √3)/6

(here alpha and beta, are just standard, you can choose otherwise if you want.)

---
now how do you know what simplex are you in ?

each lattice cell looks like this:

![skew vs unskew](image-2.png)
```

if x0 > y0:
    i1 = 1; j1 = 0 # lower-right triangle
else:
    i1 = 0; j1 = 1 # upper-left triangle

```
( basically :) )

---
next we compute the contributions from the 3 corners.

1. compute displacement vector:
```
xn​=x0​−Δxn​, yn​=y0​−Δyn​

basically:

d_vector = (x_point − x_corner, y_point − y_corner)
```
2. compute attenuation:

```
t = 0.5 - x_n^2 - y_n^2
```
3. if t < 0: contribution = 0.
4. otherwise:
    - pick gradient vector for that corner.
    - compute ramp: `ramp = g_x.n_x + g_y.n_y`
    - multiply with attenutation: ` contrib = (t^4) . ramp `

### example:
![alt text](image-3.png)

---
now unlike perlin/simplex random grandient choices, opensimplex uses a special set of gradients to reduce directional artifacts:

there are 256 or 512 gradient entries;

The gradients are chosen so the distribution is isotropic (even).
`index=perm[i+perm[j]]`
perm is a randomiser (i -> random(0-256/512)(i)).

then: 
```
gx, gy = gradients2D[index % len(gradients2D)]
```

---

so the algorithm becomes:

```
noise2D(x, y):
    # Skew the input
    s = (x + y) * alpha
    i = floor(x + s)
    j = floor(y + s)

    # Unskew the cell origin
    t = (i + j) * beta
    X0 = i - t
    Y0 = j - t
    x0 = x - X0
    y0 = y - Y0

    # Choose triangle
    if x0 > y0:
        i1 = 1; j1 = 0   # lower triangle
    else:
        i1 = 0; j1 = 1   # upper triangle

    # Corner positions relative to input
    x1 = x0 - i1 + beta
    y1 = y0 - j1 + beta
    x2 = x0 - 1 + 2*beta
    y2 = y0 - 1 + 2*beta

    # Get gradient indices from permutations
    gi0 = perm[i + perm[j]]
    gi1 = perm[i + i1 + perm[j + j1]]
    gi2 = perm[i + 1 + perm[j + 1]]

    # Compute corner contributions
    contrib0 = atten_dot(x0, y0, gi0)
    contrib1 = atten_dot(x1, y1, gi1)
    contrib2 = atten_dot(x2, y2, gi2)

    return 70 * (contrib0 + contrib1 + contrib2)

atten_dot(x, y, gi):
    t = 0.5 - x*x - y*y
    if t < 0: return 0
    t4 = t*t*t*t
    gx, gy = gradient[gi]
    return t4 * (gx*x + gy*y)
```

this is the original OpenSimplex,
but later versions exist:

let me introduce OpenSimplex2:

## OpenSimplex2s:

the main difference between this and what we already mentioned is that:

### 1. different skew/unskew constants:

```
    private static final double ROOT2OVER2 = 0.7071067811865476;

    // Rotation constants for OpenSimplex2S
    private static final double R2 = 0.5 * (Math.sqrt(3.0) - 1.0);
    private static final double R2_INV = (3.0 - Math.sqrt(3.0)) / 6.0;
```

### 2. OpenSimplex2 uses a larger, more angle-uniform gradient table (taken from the 3D/4D improvements).

we are using:
```
    private static final double[] grad2 = new double[] {
            0.130526192220052,  0.99144486137381,
            0.382683432365090,  0.923879532511287,
            0.608761429008721,  0.793353340291235,
            0.793353340291235,  0.608761429008721,
            0.923879532511287,  0.382683432365090,
            0.991444861373810,  0.130526192220051,
            0.991444861373810, -0.130526192220051,
            0.923879532511287, -0.382683432365090,
            0.793353340291235, -0.608761429008721,
            0.608761429008721, -0.793353340291235,
            0.382683432365090, -0.923879532511287,
            0.130526192220052, -0.991444861373810,
           -0.130526192220052, -0.991444861373810,
           -0.382683432365090, -0.923879532511287,
           -0.608761429008721, -0.793353340291235,
           -0.793353340291235, -0.608761429008721,
           -0.923879532511287, -0.382683432365090,
           -0.991444861373810, -0.130526192220052,
           -0.991444861373810,  0.130526192220051,
           -0.923879532511287,  0.382683432365090,
           -0.793353340291235,  0.608761429008721,
           -0.608761429008721,  0.793353340291235,
           -0.382683432365090,  0.923879532511287,
           -0.130526192220052,  0.991444861373810
    };
```

### 3. we evaluate all 4 points:

in SuperSimplex (OpenSimplex2s), we evaluate all 4 points:
```
    // We evaluate **4 points** (SuperSimplex)
    value += vertex(xsb, ysb, dx0, dy0);
    value += vertex(xsb + 1, ysb, dx0 - 1 + R2_INV, dy0 + R2_INV);
    value += vertex(xsb, ysb + 1, dx0 + R2_INV, dy0 - 1 + R2_INV);
    value += vertex(xsb + 1, ysb + 1, dx0 - 1 + 2*R2_INV, dy0 - 1 + 2*R2_INV);
```

so finally the code becomes:
[Noise.java](./src/main/java/terrain/Noise.java)

## Generate the noise map:

### step 1: Generate the base noise: done 
### step 2: apply octaves:

Realistic terrain isn’t just smooth hills; it has layers of detail.
Each octave is another layer of noise added to the base:
```
total_noise = octave1 + octave2 + octave3 + ...
```
For each octave:
- Increase frequency: the features get smaller (finer details) → controlled by lacunarity.
- Decrease amplitude: the new layer contributes less to overall height → controlled by persistence.


### step 3: normalise
After summing octaves, noiseHeight may go outside [0,1].
Normalize to [0,1]:
```
normalized = (noiseHeight + 1) / 2.0;
```

![example](image-4.png)

check the code in [NoiseMapGenerator.java](/src/main/java/terrain/NoiseMapGenerator.java)

## Analyse the terrain:

### 1. Slope:

the slope can be simply determined using :
#### $ slope = \sqrt{x^2+y^2}  $

where dx = $ (height[x + 1][y] - height[x - 1][y])/2 $
... and dy = $ (height[x][y + 1] - height[x][y - 1])/2 $

### 2. flatness:

#### $ flatness = 1 - (slope / maxSlope)$

### 3. Terrain Categorisation:

this can be done using predefined base heights:

in our case we are using:
```
if (h < waterLevel){
    terrain[x][y] = 0; // water
}
else if (h < hillLevel){
    terrain[x][y] = 1; // plains
}
else if (h < mountainLevel){
    terrain[x][y] = 2; // hill
}
else{
    terrain[x][y] = 3; // mountain
}
```

#### Improvement Strategy: Transition Zones

**Problem**: Hardcoded height thresholds create abrupt, unnatural terrain boundaries.

**Solution**: Introduce smooth **transition zones** between terrain types to create more realistic biome blending:

```
Terrain Types:
- 0 = Water (deep)
- 1 = Plains (flat low-altitude)
- 2 = Hills (mid-altitude or steep)
- 3 = Mountains (high-altitude)

Transition Zone Width: 0.05 (5% of height range)

Example logic:
if (h < waterLevel):
    terrain = 0
else if (waterLevel ≤ h < waterLevel + transition):
    terrain = 0 → 1 blend  (shallow water/beaches)
else if (waterLevel + transition ≤ h < hillLevel):
    terrain = 1 (plains)
else if (hillLevel ≤ h < hillLevel + transition):
    terrain = 1 → 2 blend  (foothills)
else if (hillLevel + transition ≤ h < mountainLevel):
    terrain = 2 (hills)
else if (mountainLevel ≤ h < mountainLevel + transition):
    terrain = 2 → 3 blend  (high peaks)
else:
    terrain = 3 (mountains)
```

**Benefits**:
- More realistic terrain progression
- Natural biome transitions (beaches, foothills, plateaus)
- Better visual aesthetic
- Improved resource distribution patterns

#### Improvement Strategy: Neighborhood Smoothing

**Problem**: Isolated single-cell terrain regions create noise and unrealistic scattered biomes (e.g., random mountain peaks surrounded by plains).

**Solution**: Apply **neighborhood-based smoothing** using local consensus voting to eliminate isolated terrain cells:

```
Algorithm:
1. Perform initial height-based categorization
2. For each cell, examine neighbors (3×3 or 5×5 radius)
3. Count terrain type votes in neighborhood
4. Assign cell to most common terrain type in vicinity
5. Repeat for 1-2 passes to strengthen coherence

Voting Example:
- Cell neighbors: [2, 2, 2, 1, X, 2, 1, 1, 3]
- Vote count: water=0, plains=3, hills=4, mountains=1
- Result: Assign cell to hills (4 votes)

Advantages:
- Eliminates isolated single-cell "islands"
- Creates coherent terrain regions
- Reduces visual noise
- Minimal performance cost at 128×128 grid
```

**Benefits**:
- ✓ Larger, more cohesive terrain regions
- ✓ Better for resource clustering (veins stay in same region)
- ✓ Improved NPC pathfinding and navigation
- ✓ More natural-looking continents

![atext](terrain_comparison.png)
![btext](terrain_statistics.png)

check it in: [TerrainAnalyser.java](./src/main/java/terrain/TerrainAnalyzer.java)


## Randomly attribute Natural Ressources:

this can be done using the informations we got from TerrainAnalyser, and a probability score combined with Simplex noise-based vein generation, that gives the possibility of a placement to be a certain ressource.

### Resource Distribution Strategy:

Instead of purely random placement, we use a **multi-scale noise vein approach** to create natural resource clusters:

#### 1. Noise-Based Vein Generation:

We leverage the existing OpenSimplex2S noise implementation (`Noise.java`) to generate resource veins at multiple scales:

- **Coarse-Scale Noise Maps**: For each resource type, we generate a noise map at a lower resolution (32×32 or 64×64) using a unique seed
- **Efficient Sampling**: Instead of evaluating noise for every cell, we pre-compute coarse noise maps once during initialization
- **Vein Patterns**: Noise gradients naturally create clusters and veins rather than scattered isolated resources

```
For each resource type:
    1. Create Noise instance with (baseSeed + resourceTypeOffset)
    2. Generate coarse noise values at lower resolution
    3. Scale/interpolate back to full 128×128 grid
    4. Sample noise values during resource placement evaluation
```

#### 2. Terrain-Modulated Thresholds:

Each resource type has biome preferences. We use **terrain-aware threshold modulation** to respect these preferences while maintaining vein patterns:

```
noiseValue = sampledNoise[x][y]  // From pre-computed coarse map

// Base threshold for resource spawn
threshold = baseThreshold

// Adjust threshold based on terrain suitability
if (terrain == favorableTerrain) {
    threshold -= 0.15    // Lower threshold in good biomes → more likely to spawn
} else if (terrain == unfavorableTerrain) {
    threshold += 0.15    // Higher threshold in bad biomes → less likely to spawn
}

if (noiseValue > threshold) {
    placeResource = true
}
```

#### 3. Resource Probability Scoring:

The final placement uses **stacked/multiplicative thresholds** to control resource density:

```
Key factors:
- **Noise Vein Pattern**: Controls base clustering and vein shapes
  - Threshold filters out noise values below a certain density
  - Only cells in high-noise regions can spawn resources
  
- **Terrain Preference**: Biome-specific modulation
  - Unfavorable terrain rejects placement entirely
  - Favorable terrain lowers threshold or increases probability
  - Neutral terrain passes through unchanged
  
- **Flatness Evaluation**: Terrain slope consideration
  - Resource-specific flatness preferences
  - Steep/flat unsuitable areas reject placement
  - Suitable flatness increases spawn probability  
```

This **stacked threshold approach**:
- Respects the vein patterns (resources only in veiny areas)
- Respects biome preferences (rejecting unsuitable terrain)
- Respects topography preferences (flatness requirements)
- Maintains tunable spawn rates (base probability)
- Provides clear behavior: setting probability to 0 fully disables resource type

#### 4. Performance Optimization:

The approach avoids expensive per-cell noise evaluation:
- ✓ Pre-compute coarse noise maps once at initialization
- ✓ O(1) sampling from cached maps during placement
- ✓ Supports real-time parameter adjustments in UI

![cmap](resource_noise_analysis.png)
![dmap](./resource_vein_comparison.png)

check out: [NaturalResourceRandomizer.java](./src/main/java/terrain/NaturalResourceRandomizer.java)

# 2. Swarm:

## Overview:

Swarm Intelligence Optimization (SIO) is used to optimally place natural resources on the generated terrain. Instead of purely random placement, we use Particle Swarm Optimization (PSO) to find locations that maximize a fitness score based on terrain characteristics and resource preferences.

## Particle Swarm Optimization (PSO):

PSO is a metaheuristic optimization algorithm inspired by the social behavior of bird flocking or fish schooling. A swarm of particles explores the search space (in our case, the 2D terrain map) to find optimal solutions.

### Core Concepts:

#### 1. Particles:

Each particle represents a potential resource placement location on the map:
- **Position**: `(x, y)` coordinates on the terrain
- **Velocity**: `(vx, vy)` direction and speed of movement
- **Personal Best**: Stores the best position this particle has found (`bestX`, `bestY`, `bestScore`)

#### 2. Global Best:

The swarm keeps track of the globally best solution found by any particle in the entire swarm.

#### 3. Update Rule:

At each iteration, particles update their velocity and position using:

```
vx_new = w * vx + c1 * r1 * (bestX - x) + c2 * r2 * (globalBestX - x)
vy_new = w * vy + c1 * r1 * (bestY - y) + c2 * r2 * (globalBestY - y)

x_new = x + vx_new
y_new = y + vy_new
```

Where:
- `w`: inertia weight (controls momentum)
- `c1`, `c2`: cognitive and social coefficients (control attraction to personal and global bests)
- `r1`, `r2`: random values in [0, 1]

This allows particles to:
- **Exploit**: Move toward known good solutions (personal and global bests)
- **Explore**: Maintain some randomness to escape local optima

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA1UAAAIhCAYAAACmO5ClAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8fJSN1AAAACXBIWXMAAA9hAAAPYQGoP6dpAAEAAElEQVR4nOydZ3gc5dWG79muVbVkWbZc5IYLxtiAKabaoddQ8wUIPSS0AElICJCEQAgQSAgkBEjoJfTeMb2aYrDBuPcmF/W6dWa+H7MzWtmSrLK7Myuf+7p8WVq9O/Nqp2ie95zzHEXXdR1BEARBEARBEAShV7jsnoAgCIIgCIIgCEI2I6JKEARBEARBEAShD4ioEgRBEARBEARB6AMiqgRBEARBEARBEPqAiCpBEARBEARBEIQ+IKJKEARBEARBEAShD4ioEgRBEARBEARB6AMiqgRBEARBEARBEPqAiCpBEARBEARBEIQ+IKJKEIR+w0MPPYSiKJ3+++CDD6yxtbW1/PjHP2bQoEEoisLxxx8PwOrVqzn66KMpLi5GURQuv/zylM/zrrvu4qGHHkr5dqPRKBdccAFDhgzB7XYzderUTseeffbZ7T4bv9/P+PHjufbaawmHw+3GvvXWWxx22GGUl5fj9/spLy9nxowZ3Hzzzdtst6WlhZtvvpnddtuNvLw8cnNzmTp1KjfeeCMtLS3d+j22ntvW/5zCjTfeyIsvvrjN6x988ME251umeeWVVzj22GMpKyvD5/NRXFzMwQcfzP/+9z9isZht8+ovPP7449x+++12T0MQBAfhsXsCgiAIqebBBx9kwoQJ27y+8847W1//+c9/5oUXXuCBBx5gzJgxFBcXA/DLX/6SL774ggceeIDBgwczZMiQlM/vrrvuYuDAgZx99tkp3e7dd9/Nf/7zH/71r3+xxx57kJeX1+X4nJwc3nvvPQDq6up44oknuP7661m8eDFPPfUUAPfccw8XXnghJ510EnfeeSfFxcWsW7eOzz77jGeffZbf/e531vY2b97MIYccwooVK7j00ku55ZZbAHjvvfe44YYbeOKJJ3jnnXcoKyvb7u+SPDencuONN3LyySdbgtxk9913Z/bs2e3Ot0yh6zrnnnsuDz30EEcddRS33XYbw4cPp6Ghgffff5+LLrqI6upqLrvssozPrT/x+OOP8/3336dl0UUQhOxERJUgCP2OXXbZhWnTpnU55vvvv2fMmDGcfvrp27y+1157bfOgnA18//335OTkcMkll3RrvMvlYp999rG+P/LII1m9ejVPP/00t912G0OHDuWmm27iwAMP5Nlnn2333jPOOANN09q9duaZZ7J48WLef/999t9/f+v1Qw89lKOPPpqZM2dy1lln8eabb/Z4btlEQUGBbXO/9dZbeeihh7juuuv44x//2O5nxx57LL/97W9Zvny5LXMTBEHoz0j6nyAIOxSrV69GURTeeecdFi1a1C41UFEUli9fzhtvvGG9vnr1agAaGxu54oorGDVqFD6fj6FDh3L55Zdvk9KmaRr/+te/mDp1Kjk5ORQVFbHPPvvw8ssvAzBy5EgWLFjAhx9+aO1j5MiRXc45HA5z1VVXtdv3xRdfTH19vTVGURTuu+8+QqGQtd3epBiaYmDNmjUA1NTUdBqtc7na/oTMmTOHWbNmcd5557UTVCb7778/5557Lm+99RZff/11j+fVGYsXL+aII44gGAwycOBALrjgAl555ZVt0u9GjhzZYWRwxowZzJgxw/o+HA7z61//mqlTp1JYWEhxcTHTp0/npZdeavc+RVFoaWnh4Ycftj5vczudpf+9/PLLTJ8+nWAwSH5+PoceeiizZ89uN+ZPf/oTiqKwYMECTj31VAoLCykrK+Pcc8+loaGhy88iFovx17/+lQkTJvCHP/yhwzGDBw9ud3xqa2u56KKLGDp0KD6fj9GjR3PNNdcQiUS2+X0vueQSHnzwQcaPH09OTg7Tpk3j888/R9d1br31VkaNGkVeXh4/+MEPthFuM2bMYJddduHjjz9mn332IScnh6FDh/KHP/wBVVXbje3pnB599FEmTpxIMBhkypQpvPrqq9v83suWLeO0005j0KBB+P1+Jk6cyL///e92Y8zj9sQTT3DNNddQXl5OQUEBhxxyCEuWLGn3u7z22musWbOmw7TUu+++mylTppCXl0d+fj4TJkzg6quv7vB4CILQf5BIlSAI/Q5VVYnH4+1eUxQFt9vNkCFDmD17NhdddBENDQ3873//A4zUwNmzZ3PCCScwZswY/va3vwEwZMgQWltbOeigg1i/fj1XX301u+66KwsWLOCPf/wj8+fP55133rEeqs4++2wee+wxzjvvPK6//np8Ph/ffPONJc5eeOEFTj75ZAoLC7nrrrsA8Pv9nf4uuq5z/PHH8+6773LVVVdxwAEH8N1333Httdcye/ZsZs+ejd/vZ/bs2fz5z3/m/ffft9LmxowZ0+PPznwYLi0tBWD69Ok899xz/OlPf+KEE05gl112we12b/O+t99+G6DLCN/xxx/Pf//7X95++2322GOP7c5l62MIhpAzxdzmzZs56KCD8Hq93HXXXZSVlfG///2v25G6johEItTW1nLFFVcwdOhQotEo77zzDieeeCIPPvggZ555JgCzZ8/mBz/4ATNnzrQETEFBQafbffzxxzn99NM57LDDeOKJJ4hEItxyyy3MmDGDd999dxshetJJJ/F///d/nHfeecyfP5+rrroKgAceeKDTfcyZM4fa2lrOP//8btWehcNhZs6cyYoVK7juuuvYdddd+fjjj7npppuYN28er732Wrvxr776KnPnzuXmm29GURSuvPJKjj76aM466yxWrlzJnXfeSUNDA7/61a846aSTmDdvXrt5bNq0iR//+Mf87ne/4/rrr+e1117jhhtuoK6ujjvvvLNXc3rttdf46quvuP7668nLy+OWW27hhBNOYMmSJYwePRqAhQsXsu+++zJixAj+/ve/M3jwYN566y0uvfRSqqurufbaa9tt8+qrr2a//fbjvvvuo7GxkSuvvJJjjz2WRYsW4Xa7ueuuu/jZz37GihUreOGFF9q998knn+Siiy7iF7/4BX/7299wuVwsX76chQsXbvd4CIKQ5eiCIAj9hAcffFAHOvzndrvbjT3ooIP0SZMmbbONiooK/eijj2732k033aS7XC79q6++avf6s88+qwP666+/ruu6rn/00Uc6oF9zzTVdznPSpEn6QQcd1K3f6c0339QB/ZZbbmn3+lNPPaUD+n//+1/rtbPOOkvPzc3t1nbNsbFYTI/FYnpVVZV+xx136Iqi6Hvuuac1bvny5fouu+xifY45OTn6wQcfrN955516NBq1xl1wwQU6oC9evLjTfS5atEgH9AsvvHC7c+vsOB588MHWuCuvvFJXFEWfN29eu/cfeuihOqC///771msVFRX6WWedtc2+DjrooC6PRTwe12OxmH7eeefpu+22W7uf5ebmdrjN999/v93+VVXVy8vL9cmTJ+uqqlrjmpqa9EGDBun77ruv9dq1117b4fG+6KKL9EAgoGua1ulcn3zySR3Q77nnnk7HJHPPPffogP7000+3e/2vf/2rDuizZs2yXgP0wYMH683NzdZrL774og7oU6dObTev22+/XQf07777znrtoIMO0gH9pZdearev888/X3e5XPqaNWt6NaeysjK9sbHRem3Tpk26y+XSb7rpJuu1ww8/XB82bJje0NDQbpuXXHKJHggE9NraWl3X247bUUcd1W7c008/rQP67NmzrdeOPvpovaKiQt+aSy65RC8qKtrmdUEQ+j+S/icIQr/jkUce4auvvmr374svvuj19l599VV22WUXpk6dSjwet/4dfvjh7dK83njjDQAuvvjiVPwaAFbUaevUtVNOOYXc3FzefffdXm+7paUFr9eL1+ultLSUyy+/nCOPPLLd6vuYMWP49ttv+fDDD7nuuus45JBD+Oqrr7jkkkuYPn36Nk6BXaHrOkC3oig5OTnbHMOvvvrKiu4BvP/++0yaNIkpU6a0e+9pp53W7Tl1xDPPPMN+++1HXl4eHo8Hr9fL/fffz6JFi3q1vSVLllBZWckZZ5zRLmUyLy+Pk046ic8//5zW1tZ27znuuOPafb/rrrsSDofZsmVLr+bQEe+99x65ubmcfPLJ7V43z7Wtz62ZM2eSm5trfT9x4kTAqMVLPqbm62YKqUl+fv42v9dpp52Gpml89NFHvZ5Tfn6+9X1ZWRmDBg2y9h0Oh3n33Xc54YQTCAaD7a7fo446inA4zOeff95umx199h39Ph2x1157UV9fz6mnnspLL71EdXX1dt8jCEL/QNL/BEHod0ycOHG7RhU9YfPmzSxfvhyv19vhz80Hp6qqKtxuN4MHD07ZvmtqavB4PFY6nomiKAwePJiamppebzsnJ8d6mPX7/VRUVHSYwuZyuTjwwAM58MADAUOMnXfeeTz11FM88MADXHTRRYwYMQKAVatWMX78+A73Z6ZADh8+fLtzc7lc2z2GNTU1jBo1apvX+/L5P//88/zoRz/ilFNO4Te/+Q2DBw/G4/Fw9913d5l6t715Ah3WppWXl6NpGnV1dQSDQev1kpKSduPMFNFQKNTpfpKPQXfnNXjw4G1E7qBBg/B4PNucW6ZDponP5+vy9a0Fd0euj+axMvfV0zlt/TmB8VmZn1NNTQ3xeJx//etf/Otf/9pmLLCN8OnNZ29yxhlnEI/HuffeeznppJPQNI0999yTG264gUMPPXS77xcEIXsRUSUI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)

### Particle Implementation:

```java
public class Particle {
    public double x, y;           // current position
    public double vx, vy;         // velocity
    public double bestX, bestY;   // personal best position
    public double bestScore;      // personal best fitness score
}
```

Particles are initialized with random positions uniformly distributed across the terrain dimensions.

## Fitness Functions:

The optimization uses composite fitness functions to evaluate location quality. Multiple `FitnessFunction` implementations evaluate different criteria:

### 1. Terrain Fitness:

Evaluates how suitable a location is based on terrain type:

```java
public class TerrainFitness implements FitnessFunction {
    // Terrain types:
    // 0 = water
    // 1 = plains
    // 2 = hills
    // 3 = mountains
    
    public double evaluate(int x, int y) {
        double score = switch(terrain[x][y]) {
            case 0 -> water_value;
            case 1 -> plains_value;
            case 2 -> hills_value;
            case 3 -> mountains_value;
            default -> 0;
        };
        
        // Adjust based on terrain flatness
        return score * (1 - (1 - flatness[x][y]) * flatness_coeff);
    }
}
```

The terrain score is modulated by **flatness coefficient** (`flatness_coeff`), allowing steep or flat areas to be penalized/rewarded based on configuration.

### 2. Resource Fitness:

Evaluates how suitable a location is for specific resource placement based on:
- **Resource presence**: Which resources are located at that position
- **Flatness preference**: Each resource type has flatness preferences

```java
public class ResourceFitness implements FitnessFunction {
    // Resource types (9 total):
    // 0 = Sedimentary Rock
    // 1 = Gemstones
    // 2 = Iron Ore
    // 3 = Coal
    // 4 = Gold Ore
    // 5 = Wood
    // 6 = Cattle Herd
    // 7 = Wolf Pack
    // 8 = Fish School
    
    public double evaluate(int x, int y) {
        double score = 0.0;
        
        // Sum values for all resources present at this location
        if (resourceTerrain[x][y][SEDIMENTARY_ROCK] == 1)
            score += sedimentaryRockValue;
        if (resourceTerrain[x][y][GEMSTONES] == 1)
            score += gemstonesValue;
        // ... etc for all resources
        
        // Boost score based on flatness
        score *= (1 + flatness[x][y]);
        
        // Apply flatness coefficient modulation
        return score * (1 - (1 - flatness[x][y]) * flatness_coeff);
    }
}
```

Each resource type has an associated score value. Locations are evaluated by summing the values of all resources present, then modulating by terrain flatness.

## Resource Placement Strategy:

The swarm optimization approach works as follows:

### 1. Candidate Solution Representation:

A candidate solution is represented as:

```java
public class ResourcePlacementSolution {
    public final int[] coords;
}
```

This simple ND coordinate represents a potential location to place/evaluate a resource grouping.

### 2. Multi-Objective Optimization:

The fitness evaluation can combine multiple fitness functions:
- **Terrain Fitness**: Ensures resources are placed on terrain types they prefer
- **Resource Fitness**: Prioritizes locations rich in valuable resources

The combined fitness allows the swarm to find locations that balance both terrain suitability and resource density.

### 3. Adaptive Placement:

By iterating the PSO algorithm, the swarm converges toward optimal placement zones where:
- Terrain type matches resource preferences
- Resource density is high
- Terrain flatness supports the settlement/extraction

This produces a more natural, efficient resource distribution compared to uniform random placement.

#### Iter 28:

![image-20251215191351705](C:\Users\hatim\AppData\Roaming\Typora\typora-user-images\image-20251215191351705.png)

#### Iter 11594:

![image-20251215191851555](C:\Users\hatim\AppData\Roaming\Typora\typora-user-images\image-20251215191851555.png)

## Advantages Over Random Placement:

1. **Terrain-Aware**: Resources are placed on suitable terrain types
2. **Clustered Distribution**: Related resources naturally group together
3. **Optimized Accessibility**: Flatness and slope considerations improve extractability
4. **Balanced Objectives**: Handles trade-offs between multiple optimization criteria
5. **Convergence**: The swarm converges toward globally good solutions rather than purely random locations
