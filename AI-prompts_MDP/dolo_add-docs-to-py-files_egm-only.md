<!-- Existing header or metadata, unchanged... -->

# Instructions for Adding Documentation to `egm.py` Without Changing Functionality

Below is a refined version of the instructions for editing `dolo/algos/egm.py`. These steps must be followed **without altering** the code's functional or structural elements—**only** documentation (inline comments and docstrings) can be added.

---

## Scope of Edits

**Allowed**:
1. Adding inline comments to currently uncommented lines (always use `# ...` at the end of lines, or after existing code).
2. Adding docstrings immediately below each function or class definition *only if* it lacks a docstring.
3. Citing relevant `.md` documentation in `docs/` or usage samples in `examples/`.

**Forbidden**:
1. Changing any existing comment text (you must leave them exactly as-is).
2. Removing or altering any lines of code (including whitespace or blank lines).
3. Adding or removing imports, renaming variables, or adjusting logic flow.
4. Changing function signatures or how they behave.
5. Modifying any code that controls functionality in `egm.py`.

---

## Step-by-Step Process

1. **PHASE 1: Inline Comments**  
   - Open `egm.py`.  
   - For each line lacking a comment, append a *short* inline comment explaining what the line does.  
   - Retain existing comments unchanged.  
   - **Do not add** or remove lines, reorder lines, or perform any other code modifications.

2. **PHASE 2: Docstrings**  
   - Immediately after each function or class definition that does **not** already have a docstring, add a suitable docstring referencing relevant docs.  
   - Provide usage examples by referencing any place in `examples/models/` or `examples/notebooks_py/` that calls the method, if applicable.  
   - As before, do **not** delete or change any line of functional code. Only insert docstrings on lines that do **not** exist (i.e., right below the `def` or `class` statement) so that the **total line count does not change**.

3. **Verification**  
   - Ensure each import statement has a short inline comment describing its purpose.  
   - Ensure each function now has a docstring.  
   - No line of the original code is removed or altered. No new functional code is introduced.

---

``` {.yaml}
age: 18
name: peter
occupations:
  - school
  - guitar
friends:
  paula: {age: 18}
```

The correspondance between the yaml definition and the resulting Python
object is very transparent. YAML mappings and lists are converted to
Python dictionaries and lists respectiveley.

!!! note
    TODO say something about YAML objects


Any model file must be syntactically correct in the Yaml sense, before
the content is analysed further. More information about the YAML syntax
can be found on the [YAML website](http://www.yaml.org/), especially
from the [language specification](http://www.yaml.org/).

Example
-------

Here is an example model contained in the file
`examples\models\rbc.yaml`

```
--8<-- "examples/models/rbc.yaml"
```


This model can be loaded using the command:

``` {.python}
model = yaml_import(`examples/models/rbc.yaml`)
```

The function `yaml_import` (cross) will raise errors until
the model satisfies basic compliance tests. . In the
following subsections, we describe the various syntactic rules prevailing
while writing yaml files.

Sections
--------

A dolo model consists in the following 4 or 5 parts:

-   a `symbols` section where all symbols used in the model
    must be defined
-   an `equations` section containing the list of equations
-   a `calibration` section providing numeric values for the
    symbols
-   a `domain` section, with the information about the
    solution domain
-   an `options` section containing additional informations
-   an `exogenous` section where exogenous shocks are defined.

These section have context dependent rules. We now review each of them in detail:

### Declaration section

This section is introduced by the `symbols`]{.title-ref}` keyword. All
symbols appearing in the model must be defined there.

Symbols must be valid Python identifiers (alphanumeric not beginning
with a number) and are case sensitive. Greek letters  are recognized. Subscripts and
superscripts can be denoted by `_` and `__`
respectively. For instance `beta_i_1__d` will be pretty
printed as $\beta_{i,1}^d$. Unicode characters are accepted too, as long as they are valid, when used within python identifiers.

!!! note

    In most modern text editor, greek characters can be typed, by entering their latex representation (like `beta`) and pressing Tab.

Symbols are sorted by type as in the following example:

``` {.yaml}
symbols:
  states: [a, b]
  controls: [u, v]
  exogenous: [e]
  parameters: [rho]
```

Note that each type of symbol is associated with a symbol list (like `[a,b]`).

!!! alert
    A common mistake consists in forgetting the commas, and use spaces only inside list. This doesn't work since the space will be ignored and the two symbols recognized as one.

The exact list of symbols to declare depends on which algorithm is meant to be used. In general, one needs to supply at least *states* (endogenous states), *exogenous* (for exogenous shocks), *controls* for 
decision variables, and *parameters* for scalar parameters, that the model can depend on.

### Declaration of equations

The `equations` section contains blocks of equations sorted by type.

Expressions follow (roughly) the Dynare conventions. Common arithmetic
operators ([+,-,\*,/,\^]{.title-ref}) are allowed with conventional
priorities as well as usual functions (`sqrt`, `log`, `exp`, `sin`, `cos`, `tan`,
`asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`).
The definitions of these functions match the definitions from the
`numpy` package.

All symbols appearing in an expression must
either be declared in the `symbols` section or be one of the predefined functions. Parameters (that are time invariant) are ntot subscripted, while all other symbol types, are variables, indexed by time. A variable `v` appear as `v[t-1]` (for $v_{t-1}$), `v[t]` (for $v_t$), 
or `v[t+1]` (for $v_t$).

<!-- Any symbol [s]{.title-ref} that is not a parameter
is assumed to be considered at date [t]{.title-ref}. Values at date
[t+1]{.title-ref} and [t-1]{.title-ref} are denoted by
[s(1)]{.title-ref} and [s(-1)]{.title-ref} respectively. -->

All equations are implicitly enclosed by the expectation operator
$E_t\left[\cdots \right]$. Consequently, the law of motion for the capital

$$k_{t+1} = (1-\delta) k_{t} +  i_{t} + \epsilon_t$$

is written (in a `transition` section) as:


``` {.yaml}
k[t] = (1-δ)*k[t-1] + i[t-1]
```

while the Euler 

$$E_t \left[ \beta \left( \frac{c_{t+1}}{c_t} + (1-\delta)+r_{t+1} \right) \right] - 1$$

would be written (in the `arbitrage` section) as:

``` {.yaml}
β*(c[t]/c[t+1])^(σ)*(1-δ+r[t+1]) - 1   # note that expectiation is operator
```

!!! note

    In Python, the exponent operator is denoted by [\*\*]{.title-ref} while
    the caret operator [\^]{.title-ref} represents bitwise XOR. In dolo
    models, we ignore this distinction and interpret both as an exponent.

!!! note

    The default evaluator in dolo preserves the evaluation order. Thus
    `(c[t+1]/c[t])^(-gamma)` is not evaluated in the same way (and is numerically
    more stable) than `c(1)^(-gamma)/c^(-gamma)`. Currently, this is not
    true for symbolically computed derivatives, as expressions are
    automatically simplified, implying that execution order is not
    guaranteed. This impacts only higher order perturbations.


An equation can consist of one expression, or two expressions separated
by [=]{.title-ref}. There are two types of equation blocks:

- __Condition blocks__: in these blocks, each equation `lhs = rhs` define the scalar value `(rhs)-(lhs)`. A list of of such equations, i.e a block, defines a multivariate function of the appearing symbols. Certain     condition blocks, can be associated with complementarity conditions separated by `⟂` (or `|`) as in  `rhs-lhs ⟂ 0 < x < 1`. In this case it is advised to omit the equal sign in order to make it easier to interpret the complementarity. Also, when complementarity conditions are used, the ordering of variables appearing in the complementarities must match the declaration order (more in section Y).
- __Definition blocks__: the differ from condition blocks in that they define a group of variables (`states` or `auxiliaries`) as a function of the right hand side.

The types of variables appearing on the right hand side depend on the
block type. The variables enumerated on the left hand-side must appear
in the declaration order.

!!! note

    In the RBC example, the `definitions` block defines variables (`y,c,rk,w`)
    that can be directly deduced from the states and the controls:

    ``` {.yaml}
    definitions:
        - y[t] = z[t]*k[t]^alpha*n[t]^(1-alpha)
        - c[t] = y[t] - i[t]
        - rk[t] = alpha*y[t]/k[t]
        - w[t] = (1-alpha)*y[t]/w[t]
    ```

Note that the declaration order matches the order in which variables
appear on the left hand side. Also, these variables are defined
recursively: `c`, `rk` and `w` depend on the value for `y`. In contrast
to the calibration block, the definition order matters. Assuming that
variables where listed as (`c,y,rk,w`) the following block would provide
incorrect result since `y` is not known when `c` is evaluated.

``` {.yaml}
definitions:
    - c[t] = y[t] - i[t]
    - y[t] = z[t]*k[t]^alpha*n[t]^(1-alpha)
    - rk[t] = alpha*y[t]/k[t]
    - w[t] = (1-alpha)*y[t]/w[t]
```

### Calibration section

The role of the calibration section consists in providing values for the
parameters and the variables. The calibration of all parameters
appearing in the equation is of course strictly necessary while the the
calibration of other types of variables is useful to define the
steady-state or an initial guess to the steady-state.

The calibrated values are also substituted in other sections, including
`exgogenous` and `options` sections. This is
particularly useful to make the covariance matrix depend on model
parameters, or to adapt the state-space to the model's calibration.

The calibration is given by an associative dictionary mapping symbols to
define with values. The values can be either a scalar or an expression.
All symbols are treated in the same way, and values can depend upon each
other as long as there is a way to resolve them recursively.

In particular, it is possible to define a parameter in order to target a
special value of an endogenous variable at the steady-state. This is
done in the RBC example where steady-state labour is targeted with
`n: 0.33` and the parameter `phi` calibrated so that the optimal labour
supply equation holds at the steady-state (`chi: w/c^sigma/n^eta`).

All symbols that are defined in the [symbols]{.title-ref} section but do not appear in the calibration section are initialized with the value [nan]{.title-ref} without issuing any warning.

!!! note

    No clear long term policy has been established yet about how to deal with undeclared symbols in the calibration section. Avoid them. TODO: reevaluate

### Domain section

The domain section contains boundaries for each endogenous state as in
the following example:

``` {.yaml}
domain:
    k: [0.5*k, 2*k]
    z: [-σ_z*3, σ_z*3]
```

!!! note

    In the above example, values can refer to the calibration dictionary.   Hence, `0.5 k` means `50%` of steady-state  `k`. Keys, are not replaced.


### Exogenous shocks specification

!!! alert
    This section is out-of-date. Syntax has changed.
    Many more shocks options are allowed. See [processes](processes.md) for a more recent description of the shocks.

    TODO: redo


The type of exogenous shock associated to a model determines the kind of decision rule, whih will be obtained by the solvers. Shocks can pertain to one of the following categories: continuous i.i.d. shocks (Normal law), continous autocorrelated process (VAR1 process) or a discrete markov chain. The type of the shock is specified using yaml type annotations (starting with exclamation mark) The exogenous shock section can refer to parameters specified in the calibration section. Here are some examples for each type of shock:

#### Normal

For Dynare and continuous-states models, one has to specifiy a
multivariate distribution of the i.i.d. process for the vector of
exogenous  variaibles (otherwise they are assumed to be constantly equal to zero). This is done
in the `exogenous` section. A gaussian distrubution (only one supported
so far), is specified by supplying the covariance matrix as a list of
list as in the following example.

``` {.yaml}
exogenous: !Normal:
    Sigma: [ [sigma_1, 0.0],
            [0.0, sigma_2] ]
```
!!! alert

    The shocks syntax is currently rather unforgiving. Normal shocks expect
    a covariance matrix (i.e. a list of list) and the keyword is `Sigma` not `sigma`.

#### Markov chains

Markov chains are constructed by providing a list of nodes and a
transition matrix.

``` {.yaml}
exogenous: !MarkovChain
    values: [[-0.01, 0.1],[0.01, 0.1]]
    transitions: [[0.9, 0.1], [0.1, 0.9]]
```

It is also possible to combine markov chains together.

``` {.yaml}
exogenous: !Product
    - !MarkovChain
        values: [[-0.01, 0.1],[0.01, 0.1]]
        transitions: [[0.9, 0.1], [0.1, 0.9]]
    - !MarkovChain
        values: [[-0.01, 0.1],[0.01, 0.1]]
        transitions: [[0.9, 0.1], [0.1, 0.9]]
```

### Options

The [options]{.title-ref} section contains all informations necessary to
solve the model. It can also contain arbitrary additional informations.
The section follows the mini-language convention, with all calibrated
values replaced by scalars and all keywords allowed.

Global solutions require the definition of an approximation space. The
lower, upper bounds and approximation orders (number of nodes in each
dimension) are defined as in the following example:

``` {.yaml}
options:
    grid: !Cartesian
        n: [10, 50]
    arbitrary_information: 42
```

```

File: docs/inspect.md
```md
Inspecting the solution
=======================

The output of most solution methods is a decision rule for the controls
as a function of the exogenous and endogenous states: `dr`. This
decision rule can be called using one of the following methods:

-   `dr.eval_s(s: array)`: function of endogenous state. Works only if
    exgogenous process is i.i.d.
-   `dr.eval_ms(m: array,s: array)`: function of exogenous and
    endogenous values. Works only if exogenous process is continuous.
-   `dr.eval_is(i: int,s: array)`: function of exognous index and
    endogenous values. Works only if some indexed discrete values are
    associated with exogenous process.

There is also a \_\_call\_\_ function, which tries to make the sensible
call based on argument types. Hence `dr(0, s)` will behave as the third
example.

Tabulating a decision rule
--------------------------

Dolo provides a convenience function to plot the values of a decision
rule against different values of a state:

![mkapi](dolo.algos.simulations.tabulate)

Stochastic simulations
----------------------

Given a model object and a corresponding decision rule, one can get a
`N` stochastic simulation for `T` periods, using the `simulate`
function. The resulting object is an 3-dimensional *DataArray*, with the
following labelled axes:

- T: date of the simulation (`range(0,T)`)
- N: index of the simulation (`range(0,N)`)
- V: variables of the model (`model.variables`)

![mkapi](dolo.algos.simulations.simulate)


Impulse response functions
--------------------------

For continuously valued exogenous shocks, one can perform an impulse
response function:

![mkapi](dolo.algos.simulations.response)


Graphing nonstochastic simulations
----------------------------------

Given one or many nonstochstic simulations of a model, obtained with
`response`, or `deterministic_solve` it is possible to quickly create an
irf for multiple variables.

![mkapi](dolo.misc.graphs.plot_irfs)

```

File: docs/finite_iteration.md
```md
Finite iteration
================

By default, dolo looks for solutions, where the time horizon of the optimizing agent is infinite. But it is possible to use it to solve finite horizon problems (only with time-iteration, so far). It requires to do two things:

- Construct a final decision rule to be used for the last period. This can be done with the `CustomDR` function. ResultingsIt is passed as an initial guess to `time_iteration`.
- Record the decision rules, obtained in each iteration.

!!! example

    ```
    from dolo import yaml_import, time_iteration, tabulate
    from dolo.numeric.decision_rule import CustomDR

    model = yaml_import("examples/models/consumption_savings.yaml")

    last_dr = CustomDR(
        {"c": "w"},
        model=model
    )

    T = 10

    result = time_iteration(model, 
        dr0=last_dr, 
        maxit=T,
        trace=True # keeps decision rules, from all iterations
    )

    # example to plot all decision rules
    from matplotlib import pyplot as plt
    for i,tr in enumerate(result.trace):
        dr = tr['dr']
        tab = tabulate(model, dr, 'w')
        plt.plot(tab['w'], tab['c'], label=f"t={T-i}")
    plt.legend(loc='upper left')
    plt.show()
    ```

    In the example above, {"c": "w"} stands for a functional identity, i.e. c(y,w) = w. It is a completely different meaning from `c: w` in the calibration section which means that the steady-state value of `c` is initialized to the steady-state value of `w`.


!!!alert

    The notation in CustomDR is not yet consistent, with the new timing conventions. In the example above it should be `c[t] = w[t]`. A commission, will be created to examine the creation of an issue, meant to coordinate the implementation of a solution.
```

File: docs/processes.md
```md
# Shocks

The type of exogenous shock associated to a model determines the kind of decision rule, which will be obtained by the solvers. Shocks can pertain to one of the following categories:

- continuous i.i.d. shocks aka distributions

- continuous auto-correlated process such as AR1

- discrete processes such as discrete markov chains


Exogenous shock processes are specified in the section `exogenous` of a yaml file.

Here are some examples for each type of shock:

## Distributions / IIDProcess

### Univariate distributions

#### IID Normal

The type of the shock is specified using yaml type annotations (starting with exclamation mark)

Normal distribution with mean mu and variance σ^2 has the probability density function

$$f(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}}
\exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)$$

A normal shock in the yaml file with mean 0.2 and standard deviation 0.1 can be declared as follows

```
!Normal:
    σ: 0.1
    μ: 0.2
```

or

```
!Normal:
    sigma: 0.1
    mu: 0.2
```

!!! note
    Greek letter 'σ' or 'sigma' (similarly 'μ' or 'mu' ) are accepted.


!!! note

      When defining shocks in a dolo model, that is in an `exogenous` section, The exogenous shock section can refer to parameters specified in the calibration section:

      ```   
      symbols:
      ...
            parameters: [alpha, beta, mu, sigma]
      ...
      calibration:
            sigma: 0.01
            mu: 0.0

      exogenous: !Normal:
            σ: sigma
            μ: mu

      ```      

#### IID LogNormal

Parametrization of a lognormal random variable Y is in terms of he mean, μ, and standard deviation, σ, of the unique normally distributed random variable X is as follows

$$f(x; \mu, \sigma) = \frac{1}{x \sqrt{2 \pi \sigma^2}}
\exp \left( - \frac{(\log(x) - \mu)^2}{2 \sigma^2} \right),
\quad x > 0$$

such that exp(X) = Y

```
exogenous: !LogNormal:
      σ: sigma
      μ: mu

```    

#### Uniform

Uniform distribution over an interval [a,b]

$$f(x; a, b) = \frac{1}{b - a}, \quad a \le x \le b$$


```
symbols:
      states: [k]
      controls: [c, d]
      exogenous: [e]
      parameters: [alpha, beta, mu, sigma, e_min, e_max]

.
.
.

exogenous: !Uniform:
      a: e_min
      b: e_max

```    

#### Beta

If X∼Gamma(α) and Y∼Gamma(β) are distributed independently, then X/(X+Y)∼Beta(α,β).

Beta distribution with shape parameters α and β has the following PDF

$$f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1} (1 x)^{\beta - 1}, \quad x \in [0, 1]$$

```
exogenous: !Beta
      α: 0.3
      β: 0.1

```    

#### Bernouilli

Binomial distribution parameterized by $p$ yields $1$ with probability $p$ and $0$ with probability $1-p$.

```
!Bernouilli
      π: 0.3
```   
### Multivariate distributions

#### Normal (multivariate)

Note the difference with `UNormal`. Parameters `Σ` (not `σ`) and `μ` take a matrix and a vector respectively as argument.
```
!Normal:
      Σ: [[0.0]]
      μ: [0.1]
```


### Mixtures

For now, mixtures are defined for i.i.d. processes only. They take an integer valued distribution (like the Bernouilli one) and a different distribution associated to each of the values.

```yaml
exogenous: !Mixture
    index: !Bernouilli
        p: 0.3
    distributions:
        0: UNormal(μ=0.0, σ=0.01)
        1: UNormal(μ=0.0, σ=0.02)
```

Mixtures are not restricted to 1d distributions, but all distributions of the mixture must have the same dimension.

!!! note

Right now, mixtures accept only distributions as values. To switch between constants, one can use a `Constant` distribution as in the following examples.

```yaml
exogenous:
    e,v: !Mixture:
        index: !Bernouilli
            p: 0.3
        distributions:
            0: Constant(μ=[0.1, 0.2])
            1: Constant(μ=[0.2, 0.3])
```

## Continuous Autoregressive Process

### AR1 / VAR1

For now, `AR1` is an alias for `VAR1`. Autocorrelation `ρ` must be a scalar (otherwise we don't know how to discretize).

```yaml
exogenous: !AR1
    rho: 0.9
    Sigma: [[σ^2]]
```


## Markov chains

Markov chains are constructed by providing a list of nodes and a
transition matrix.

```yaml
exogenous: !MarkovChain
    values: [[-0.01, 0.1],[0.01, 0.1]]
    transitions: [[0.9, 0.1], [0.1, 0.9]]
```


## Product

We can also specify more than one process. For instance if we want to combine a VAR1 and an Normal Process we use the tag Product and write:

```
exogenous: !Product

    - !VAR1
         rho: 0.75
         Sigma:  [[0.015^2, -0.05], [-0.05, 0.012]]

    -  !Normal:
          σ: sigma
          μ: mu
```

!!! note

      Note that another syntax is accepted, in the specific context of a dolo exogenous section. It keeps the Product operator implicit. Suppose a dolo model has $r,w,e$ as exogenous shocks. It is possible to list several shocks for each variable as in the following example:

      ```
      symbols:
            ...
            exogenous: [r,w,e]

      exogenous:
          r,w: !VAR1
               rho: 0.75
               Sigma: [[0.015^2, -0.05], [-0.05, 0.012]]

          e  !Normal:
                σ: sigma
                μ: mu
      ```

      In this case we define several shocks for several variables (or combinations thereof).

## Conditional processes

Support is very limited for now. It is possible to define markov chains, whose transitions (not the values) depend on the output of another process.

```
exogenous: !Conditional
    condition: !UNormal
        mu: 0.0
        sigma: 0.2
    type: Markov
    arguments: !Function
        arguments: [x]
        value:
          states: [0.1, 0.2]
          transitions: !Matrix
              [[1-0.1-x, 0.1+x],
               [0.5,       0.5]]

```

!!! note

      The plan is to replace the clean and explicit but somewhat tedious syntax above by the following (where dependence is detected automatically):

      ```
      exogenous:
          x: !UNormal
              mu: 0.0
              sigma: 0.2
          y: !Markov
                states: [0.1, 0.2]
                transitions: !Matrix
                    [[1-0.1-x, 0.1+x],
                     [0.5,       0.5]]

      ```


## Discretization methods for continous shocks

To solve a non-linear model with a given exogenous process, one can apply different types of procedures to discretize the continous process:

| Type | Distribution | Discretization procedure             |
|--------------|--------------|-----------------------------------|
|Univariate iid| UNormal(μ, σ)| Equiprobable, Gauss-Hermite Nodes |
|Univariate iid| LogNormal(μ, σ) |Equiprobable |
|Univariate iid| Uniform(a, b ) |Equiprobable|
|Univariate iid| Beta(α, β)   |Equiprobable |
|Univariate iid| Beta(α, β)   |Equiprobable |
| VAR1 |   |Generalized Discretization Method (GDP), Markov Chain |

!!! note
    Here we can define shortly each method. Then perhaps link to a jupyter notebook as discussed: Conditional on the discretization approach, present the results of the corresponding method solutions and simulations. Discuss further discretization methods and related dolo objects.

```

File: docs/value_iteration.md
```md
Value function iteration
========================

![mkapi](dolo.algos.value_iteration.evaluate_policy)

![mkapi](dolo.algos.value_iteration.value_iteration)

![mkapi](dolo.algos.results.ValueIterationResult)
```

File: docs/perturbation.md
```md
Perturbation
============

![mkapi](dolo.algos.perturbation.perturb)

```

File: docs/model_specification.md
```md
Model Specification
===================

Variables
--------------

### Variable types

The following types of variables can be used in models:

> -   `exogenous` (`m`) (can be autocorrelated)
> -   `states` (`s`)
> -   `controls` (`x`)
> -   `rewards` (`r`)
> -   `values` (`v`)
> -   `expectations` (`z`)
> -   `parameters` (`p`)

Symbol types that are present in a model are always listed in that
order.

### State-space

The unknown vector of controls $x$ is a function $\varphi$ of the
states, both exogenous ($m$) and endogenous ($s$) In
general we have:

$$x = \varphi(m, s)$$

In case the exogenous process is iid, dolo looks for a decision rule $x=\varphi(s)$.

!!! info

    This fact must be kept in mind when designing a model.
    TODO: explain how one can get the RBC wrong...

The function $\varphi$ is typically approximated by the solution
algorithm. It can be either a Taylor expansion, or an intepolating
object (splines, smolyak). In both cases, it behaves like a numpy gufunc
and can be called on a vector or a list of points:

``` {.python}
# for an iid model
dr = perturb(model)
m0, s0 = model.calibration['exogenous', 'states']
dr(m0, s0)                               # evaluates on a vector
dr(m0, s0[None,:].repeat(10, axis=0) )   # works on a list of points too
```


Equations
--------------


### Valid equations

The various equations that can be defined using these symbol types is
summarized in the following table. They are also reviewed below with
more details.


| Function                              | Standard name     | Short name | Definition                    |
| ------------------------------------- | ----------------- | ---------- | ----------------------------- |
| Transitions                           | `transition`      | `g`        | `s = g(m(-1), s(-1),x(-1),m)` |
| Lower bound                           | `controls_lb`     | `lb`       | `x_lb = lb(m, s)`             |
| Upper bound                           | `controls_ub`     | `ub`       | `x_ub = ub(m, s)`             |
| Utility                               | `utility`         | `u`        | `r = u(m,s,x)`                |
| Value updating                        | `alue_updating`   | `v`        | `w = v(s,x,v,s(1),x(1),w(1))` |
| Arbitrage                             | `arbitrage`       | `f`        | `0=f(m,s,x,m(1),s(1),x(1))`   |
| Expectations                          | `expectation`     | `h`        | `z=h(s(1),x(1))`              |
| Generalized  expectations             | `expectation_2`   | `h_2`      | `z=h_2(s,x,m(1),s(1),x(1))`   |
| Arbitrage  (explicit    expectations) | `arbitrage_2`     | `f_2`      | `0=f_2(s,x,z)`                |
| Direct response                       | `direct_response` | `d`        | `x=d(s,z)`                    |

When present these functions can be accessed from the `model.functions`
dictionary by using the standard name. For instance to compute the
auxiliary variables at the steady-state one can compute:

``` {.python}
# recover steady-state values
e = model.calibration['exogenous']
s = model.calibration['states']
x = model.calibration['controls']
p = model.calibration['parameters']

# compute the vector of auxiliary variables
a = model.functions['auxiliary']
y = a(e,s,x,p)

# it should correspond to the calibrated values:
calib_y = model.calibration['auxiliaries']
assert( abs(y - calib_y).max() < 0.0000001 )
```

### Equation Types

#### Transitions

    - name: `transition`
    - short name: `g`

Transitions are given by a function $g$ such that at all times:

$$s_t = g(m_{t-1}, s_{t-1}, x_{t-1}, m_t)$$

where $m_t$ is a vector of exogenous shocks

!!! example

    In the RBC model, the vector of states is $s_t=(a_t,k_t)$. The
    transitions are:

    $$\begin{eqnarray}a_t &= & \rho a_{t-1} + \epsilon_t\\
    k_t & = & (1-\delta) k_{t-1} + i_{t-1}\end{eqnarray}$$

    The yaml file is amended with:

    ``` {.yaml}
    symbols:
        states: [a,k]
        controls: [i]
        shocks: [ϵ]
        ...
    equations:
        transition:
            a[t] = rho*a[t-1] + ϵ[t]
            k = k[t-1]*(1-δ) + i[t-1]
    ```

Note that the transitions are given in the declaration order.

#### Auxiliary variables

    - name: `auxiliary`
    - short name: `a`

In order to reduce the number of variables, it is useful to define
auxiliary variables $y_t$ using a function $a$ such that:

$$y_t = a(m_t, s_t, x_t)$$

When they appear in an equation they are automatically substituted by
the corresponding expression in $m_t$, $s_t$ and $x_t$. Note that auxiliary
variables are not explicitely listed in the following definition.
Implicitly, wherever states and controls are allowed with the same date
in an equation type, then auxiliary variable are also allowed as long as the variables, they depend on are allowed. 

Auxiliary variables are defined in a special `definitions` block.

!!! example

    In the RBC model, three auxiliary variables are defined
    $y_t, c_t, r_{k,t}$ and $w_t$. They are a closed form function of
    $a_t, k_t, i_t, n_t$. Defining these variables speeds up computation
    since they are don't need to be solved for or interpolated.

#### Utility function and Bellman equation

    - name: `utility`
    - short name: `u`

The (separable) value equation defines the value $v_t$ of a given policy
as:

$$v_t = u(m_t, s_t,x_t) + \beta E_t \left[ v_{t+1} \right]$$

This gives rise to the Bellman equation:

> $$v_t = \max_{x_t} \left( u(m_t,s_t,x_t) + \beta E_t \left[ v_{t+1} \right] \right)$$

These two equations are characterized by the reward function $u$ and the
discount rate $\beta$. Function $u$ defines the vector of symbols
`rewards`. Since the definition of $u$ alone is not sufficient, the
parameter used for the discount factor must be given to routines that
compute the value. Several values can be computed at once, if $U$ is a
vector function and $\beta$ a vector of discount factors, but in that
case in cannot be used to solve for the Bellman equation.

!!! example

    Our RBC example defines the value as
    $v_t = \frac{(c_t)^{1-\gamma}}{1-\gamma} + \beta E_t v_{t+1}$. This
    information is coded using: \#\# TODO add labour to utility

    ``` {.yaml}
    symbols:
        ...
        rewards: [r]

    equations:
        ...
        utility:
            - r[t] = c[t]^(1-γ)/(1-γ)

    calibration:
        ...
        beta: 0.96   # beta is the default name of the discount
    ```


#### Value

    - name: `value`
    - short name: `w`

A more general updating equation can be useful to express non-separable
utilities or prices. the vector of (generalized) values $v^{*}$ are
defined by a function `w` such that:

$$v_t = w(m_t,s_t,x_t,v_t,m_{t+1},s_{t+1},x_{t+1},v_{t+1})$$

As in the separable case, this function can either be used to compute
the value of a given policy $x=\varphi()$ or in order solve the
generalized Bellman equation:

$$v_t = \max_{x_t} \left( w(m_t,s_t,x_t,v_t,m_{t+1},s_{t+1},x_{t+1},v_{t+1}) \right)$$

!!! example

    Instead of defining the rewards of the RBC example, one can instead
    define a value updating equation instead:

    ``` {.yaml}
    symbols:
        ...
        values: [v]

    equations:
        ...
        value:
            - v[t] = c[t]^(1-γ)/(1-γ)*(1-n[t]) + β*v[t+1]
    ```

#### Boundaries

    - name: `controls_lb` and `controls_ub`
    - short name: `lb` and `ub`

The optimal controls must also satisfy bounds that are function of
states. There are two functions $\underline{b}()$ and $\overline{b}()$
such that:

$$\underline{b}(m_t, s_t) \leq x_t \leq \overline{b}(m_t, /s_t)$$

!!! example

    In our formulation of the RBC model we have excluded negative
    investment, implying $i_t \geq 0$. On the other hand, labour cannot be
    negative so that we add lower bounds to the model:

    ``` {.yaml}
    equations:
        ...
        controls_lb:
            i = 0
            n = 0
    ```

    TODO: this makes no sense.

    Specifying the lower bound on labour actually has no effect since agents
    endogeneously choose to work a positive amount of time in order to
    produce some consumption goods. As for upper bounds, it is not necessary
    to impose some: the maximum amount of investment is limited by the Inada
    conditions on consumption. As for labour `n`, it can be arbitrarily
    large without creating any paradox. Thus the upper bounds are omitted
    (and internally treated as infinite values).

#### Euler equation

    - name: `arbitrage` (`equilibrium`)
    - short name: `f`

A general formulation of the Euler equation is:

$$0 = E_t \left[ f(m_t, s_t, x_t, m_{t+1}, s_{t+1}, x_{t+1}) \right]$$

Note that the Euler equation and the boundaries interact via
"complementarity conditions". Evaluated at one given state, with the
vector of controls $x=(x_1, ..., x_i, ..., x_{n_x})$, the Euler equation
gives us the residuals $r=(f_1, ..., f_i, ...,
f_{n_x})$. Suppose that the $i$-th control $x_i$ is supposed to lie in
the interval $[ \underline{b}_i, \overline{b}_i ]$. Then one of the
following conditions must be true:

-   $f_i$ = 0
-   $f_i<0$ and $x_i=\overline{b}_i$
-   $f_i>0$ and $x_i=\underline{b}_i$

By definition, this set of conditions is denoted by:

-   $f_i = 0 \perp \underline{b}_i \leq x_i \leq \overline{b}_i$

These notations extend to a vector setting so that the Euler equations
can also be written:

$$0 = E_t \left[ f(m_t, s_t, x_t, m_{t+1}, s_{t+1}, x_{t+1}) \right] \perp \underline{b}(m_t, s_t) \leq x_t \leq \overline{b}(m_t, s_t)$$

Specifying the boundaries together with Euler equation, or providing
them separately is exactly equivalent. In any case, when the boundaries
are finite and occasionally binding, some attention should be devoted to
write the Euler equations in a consistent manner. In particular, note
that the Euler equations are order-sensitive.

The Euler conditions, together with the complementarity conditions
typically often come from Kuhn-Tucker conditions associated with the
Bellman problem, but that is not true in general.

!!! example

    The RBC model has two Euler equations associated with investment and
    labour supply respectively. They are added to the model as:

    ``` {.yaml}
    arbitrage:
        - 1 - beta*(c[t]/c[t+1])^(sigma)*(1-delta+rk[t+1])  ⟂ 0 <= i[t] <= inf
        - w - chi*n[t]^eta*c[t]^sigma                       ⟂ 0 <= n[t] <= inf
    ```

    Putting the complementarity conditions close to the Euler equations,
    instead of entering them as separate equations, helps to check the sign
    of the Euler residuals when constraints are binding. Here, when
    investment is less desirable, the first expression becomes bigger. When
    the representative is prevented to invest less due to the constraint
    (i.e. $i_t=0$), the expression is then *positive* consistently with the
    complementarity conventions.


#### Expectations

    - name: `expectation`
    - short name: `h`

The vector of explicit expectations $z_t$ is defined by a function $h$
such that:

$$z_t = E_t \left[ h(m_{t+1}, s_{t+1},x_{t+1}) \right]$$

!!! example

    In the RBC example, one can define. the expected value tomorrow of one additional unit invested tomorrow:

    $$m_t=\beta c_{t+1}^{-\sigma}*(1-\delta+r_{k,t+1})$$

     It is a pure expectational variable in the sense that it is solely determined by future states and decisions. In the model file, it would be defined as:

    ```yaml

    symbols:
      ...
      expectations: [z]

    equations:
      expectations:
        - z = beta*(c[t+1])^(-sigma)*(1-delta+rk[t+1])
    ```

#### Generalized expectations

    - name: `expectation_2`
    - short name: `h_2`

The vector of generalized explicit expectations $z_t$ is defined by a
function $h^{\star}$ such that:

$$z_t = E_t \left[ h^{\star}(m_t, s_t,x_t,m_{t+1},s_{t+1},x_{t+1}) \right]$$

#### Euler equation with expectations

    - name: `arbitrage_2` (`equilibrium_2`)
    - short name: `f_2`

If expectations are defined using one of the two preceding definitions,
the Euler equation can be rewritten as:

$$0 = f(m_t, s_t, x_t, z_t) \perp \underline{b}(m_t, s_t) \leq x_t \leq \overline{b}(m_t, s_t)$$

!!! note

    Given the definition of the expectation variable $m_t$, today's
    consumption is given by: $c_t = z_t^{-\frac{1}{sigma}}$ so the Euler
    equations are rewritten as:

    ``` {.yaml}
    arbitrage_2:
        - 1 - beta*(c[t])^(sigma)/m[t]   | 0 <= i[t] <= inf
        - w[t] - chi*n[t]^eta*c[t]^sigma    | 0 <= n[t] <= inf
    ```

    Note the type of the arbitrage equation (`arbitrage_2` instead of
    `arbitrage`).

    However $c_t$ is not a control itself, but the controls $i_t, n_t$ can be easily deduced:

    $$\begin{eqnarray}
        n_t & =& ((1-\alpha) z_t k_t^\alpha \frac{m_t}{\chi})^{\frac{1}{\eta+\alpha}} \\
        i_t & = & z_t k_t^{\alpha} n_t^{1-\alpha} - (m_t)^{-\frac{1}{\sigma}}
    \end{eqnarray}$$

    This translates into the following YAML code:

    ``` {.yaml}
    arbitrage_2:
        - n[t] = ((1-alpha)*a[t]*k[t]^alpha*m[t]/chi)^(1/(eta+alpha))
        - i[t] = z[t]*k[t]^alpha*n[t]^(1-alpha) - m[t]^(-1/sigma)
    ```

#### Direct response function

    - name: `direct_response`
    - short name: `d`

In some simple cases, there a function $d()$ giving an explicit
definition of the controls:

$$x_t = d(m_t, s_t, z_t)$$

Compared to the preceding Euler equation, this formulation saves
computational time by removing the need to solve a nonlinear system to
recover the controls implicitly defined by the Euler equation.
```

File: docs/time_iteration.md
```md
Time iteration
==============

We consider a model with the form:

$$\begin{aligned}
s_t & = & g\left(m_{t-1}, s_{t-1}, x_{t-1}, m_t \right) \\
0   & = & E_t \left[ f\left(m_t, s_{t}, x_{t}, m_{t+1}, s_{t+1}, x_{t+1} \right) \right]
\end{aligned}$$

where $g$ is the state transition function, and $f$ is the arbitrage equation.

The time iteration algorithm consists in approximating the optimal controls as a function $\varphi$ of exogenous and endogenous controls $x_t = \varphi(m_t,s_t)$.

- At step $n$, the current guess for the control, $x(s_t) = \varphi^n(m_t, s_t)$, serves as the control being exercised next period :
    - Taking $\varphi^n$ as the initial guess, find the current period's controls $\varphi^{n+1}(m_t,s_t)$  for any $(m_t,s_t)$ by solving the arbitrage equation :
$0 = E_t \left[ f\left(m_t, s_{t}, \varphi^{n+1}(m_t, s_t), g(m_t, s_t, \varphi^{n+1}(m_t, s_t), m_{t+1}), \varphi^{n}(m_{t+1},g(m_t, s_t, \varphi^{n+1}(m_t, s_t), m_{t+1})) \right) \right]$
- Repeat until $\eta_{n+1} = \max_{m,s}\left |\varphi^{n+1}(m,s) - \varphi^{n}(m,s) \right|$ is smaller than prespecified criterium $\tau_{η}$

![mkapi](dolo.algos.time_iteration.time_iteration)

![mkapi](dolo.algos.results.TimeIterationResult)


```

File: docs/index.md
```md
# Dolo


## Introduction

Dolo is a tool to describe and solve economic models. It provides a simple classification scheme to describe many types of models, allows to write the models as simple text files and compiles these files into efficient Python objects representing them. It also provides many reference solution algorithms to find the solution of these models under rational expectations.

Dolo understand several types of nonlinear models with occasionnally binding constraints (with or without exogenous discrete shocks), as well as local pertubations models, like Dynare. It is a very adequate tool to study zero-lower bound issues, or sudden-stop problems, for instance.

Sophisticated solution routines are available: local perturbations up to third order, perfect foresight solution, policy iteration, value iteration. Most of these solutions are either parallelized or vectorized. They are written in pure Python, and can easily be inspected or adapted.

Thanks to the simple and consistent Python API for models, it is possible to write models in pure Python, or to implement other solution algorithms on top it.


## Frequently Asked Questions

No question was ever asked. Certainly because it's all very clear.

```

File: docs/perfect_foresight.md
```md
Perfect foresight
=================

Consider a series for the exogenous process $(m_t)_{0 \leq t \leq T}$ given exogenously.

The perfect foresight problem consists in finding the path of optimal
controls $(x_t)_{0 \leq t \leq T}$ and corresponding states $(s_t)_{0 \leq t \leq T}$ such that:

$$\begin{aligned}
s_t & = & g\left(m_{t-1}, s_{t-1}, x_{t-1}, m_t \right) & \\
0 & = & f\left(m_{t}, s_{t}, x_{t}, m_{t+1}, s_{t+1}, x_{t+1}\right) & \ \perp \ \underline{u} <= x_t <= \overline{u}
\end{aligned}$$

Special conditions apply for the initial state and controls. Initial
state (${m_0}, {s_0})$ is given exogenously. Final states and controls are
determined by assuming the exogenous process satisfies $m_t=m_T$ for all
$t\geq T$ and optimality conditions are satisfied in the last period:

$$f(m_T, s_T, x_T, m_T, s_T, x_T) \perp \underline{u} \leq x_T \leq \overline{u}$$

We assume that $\underline{u}$ and $\overline{u}$ are constants. This is
not a big restriction since the model can always be reformulated in
order to meet that constraint, by adding more equations.

The stacked system of equations satisfied by the solution is:

| Transitions | Arbitrage | 
|-------------|------------|
| $s_0$ exogenous |  $f(m_0, s_0, x_0, m_1, s_1, x_1) \perp \underline{u} <= x_0 <= \overline{u}$ |
| $s_1 = g(m_0, s_0, x_0, m_1)$ | $f(s_1, x_1, s_2, x_2) \perp \underline{u} <= x_1 <= \overline{u}$ |
| .... | ... |
| $s_T = g(m_{T-1}, s_{T-1}, x_{T-1}, m_T)$ | $f(m_T, s_T, x_T, m_T, s_T, x_T) \perp \underline{u} <= x_T <= \overline{u}$ |

The system is solved using a nonlinear solver.


![mkapi](dolo.algos.perfect_foresight.deterministic_solve)


```

File: docs/model_api.md
```md
# Model API

For numerical purposes, models are essentially represented as a set of symbols, calibration and functions representing the various equation
types of the model. This data is held in a `Model` object whose API is described in this chapter. Models are usually created by writing a Yaml files as described in the the previous chapter, but as we will
see below, they can also be written directly as long as they satisfy the requirements detailed below.

## Model Object


As previously, let's consider, the Real Business Cycle example, from the introduction. The model object can be created using the yaml file:

``` {.python}
model = yaml_import('models/rbc.yaml')
```

The object contains few meta-data:

``` {.yaml}
display( model.name )  # -> "Real Business Cycles"
display( model.infos )   # -> {"name": "Real Business Cycle",  "filename": "examples/models/rbc.yaml",  "type": "dtcc"}
```

## Calibration


Each models stores calibration information as `model.calibration`. It is a special dictionary-like object,  which contains calibration information, that is values for parameters and initial values (or steady-state) for all other variables of the model.

It is possible to retrieve one or several variables calibrations:

``` python
display( model.calibration['k'] ) #  ->  2.9
display( model.calibration['k', 'delta']  #  -> [2.9, 0.08]
```

When a key coresponds to one of the symbols group, one gets one or several vectors of variables instead:

```python
model.calibration['states'] # - > np.array([2.9, 0]) (values of states [z, k])
model.calibration['states', 'controls'] # -> [np.array([2.9, 0]), np.array([0.29, 1.0])]
```


To get regular dictionary mapping states groups and vectors, one can use the attributed `.grouped`
The values are vectors (1d numpy arrays) of values for each symbol group. For instance the following code will print the calibrated values of the parameters:

```python
for (variable_group, variables) in model.calibration.items():
    print(variables_group, variables)
```

In order to get a ``(key,values)`` of all the values of the model, one can call ``model.calibration.flat``.

```python
for (variable_group, variables) in model.calibration.items():
    print(variables_group, variables)
```


!!! note

    The calibration object can contain values that are not symbols of the model. These values can be used to calibrate model parameters
    and are also evaluated in the other yaml sections, using the supplied value.


One uses the `model.set_calibration()` routine to change the calibration of the model.  This one takes either a dict as an argument, or a set of keyword arguments. Both calls are valid:

```python
model.set_calibration( {'delta':0.01} )
model.set_calibration( {'i': 'delta*k'} )
model.set_calibration( delta=0.08, k=2.8 )
```

This method also understands symbolic expressions (as string) which makes it possible to define symbols as a function of other symbols:

```python
model.set_calibration(beta='1/(1+delta)')
print(model.get_calibration('beta'))   # -> nan

model.set_calibration(delta=0.04)
print(model.get_calibration(['beta', 'delta'])) # -> [0.96, 0.04]
```

Under the hood, the method stores the symbolic relations between symbols. It is precisely equivalent to use the ``set_calibration`` method
or to change the values in the yaml files. In particular, the calibration order is irrelevant as long as all parameters can be deduced one from another.


## Functions

A model of a specific type can feature various kinds of functions. For instance, a continuous-states-continuous-controls models, solved by iterating on the Euler equations may feature a transition equation $g$ and an arbitrage equation $f$. Their signature is respectively $s_t=g(m_{t-1},s_{t-1},x_{t-1},m_t)$ and $E_t[f(m_t,s_t,x_t,s_{t+1},x_{t+1},m_{t+1})]$, where $s_t$, $x_t$ and $m_t$ respectively represent a vector of states, controls and exogenous shock. Implicitly, all functions are also assumed to depend on the vector of parameters $p$.

These functions can be accessed by their type in the model.functions dictionary:

``` python
g = model.functions['transition']
f = model.functions['arbitrage']
```

Let's call the arbitrage function on the steady-state value, to see the residuals at the deterministic steady-state:

``` python
m = model.calibration["exogenous"]
s = model.calibration["states"]
x = model.calibration["controls"]
p = model.calibration["parameters"]
res = f(m,s,x,m,s,x,p)
display(res)
```

The output (`res`) is two element vector, representing the residuals of the two arbitrage equations at the steady-state. It should be full of zero. Is it ? Great !

By inspecting the arbitrage function ( `f?` ), one can see that its call api is:

```python
f(m,s,x,M,S,X,p,diff=False,out=None)
```

Since `m`, `s` and `x` are the short names for exogenous shocks, states and controls, their values at date $t+1$ is denoted with `S` and `X`. This simple convention prevails in most of dolo source code: when possible, vectors at date `t` are denoted with lowercase, while future vectors are with upper case. We have already commented the presence of the parameter vector `p`.
Now, the generated functions also gives the option to perform in place computations, when an output vector is given:

```python
out = numpy.ones(2)
f(m,s,x,m,s,x,p,out)   # out now contains zeros
```

It is also possible to compute derivatives of the function by setting ``diff=True``. In that case, the residual and jacobians with respect to the various arguments are returned as a list:

```python
r, r_m, r_s, r_x, r_M, r_S, r_X = f(m,s,x,m,s,x,p,diff=True)
```

Since there are two states and two controls, the variables ``r_s, r_x, r_S, r_X`` are all 2 by 2 matrices.

The generated functions also allow for efficient vectorized evaluation. In order to evaluate the residuals :math:`N` times, one needs to supply matrix arguments, instead of vectors, so that each line corresponds to one value to evaluate as in the following example:

```python
N = 10000

vec_m = m[None,:].repeat(N, axis=0) # we repeat each line N times
vec_s = s[None,:].repeat(N, axis=0) # we repeat each line N times
vec_x = x[None,:].repeat(N, axis=0)
vec_X = X[None,:].repeat(N, axis=0)
vec_p = p[None,:].repeat(N, axis=0)

# actually, except for vec_s, the function repeat is not need since broadcast rules apply
vec_s[:,0] = linspace(2,4,N)   # we provide various guesses for the steady-state capital
vec_S = vec_s

out = f(vec_m, vec_s,vec_x,vec_M, vec_S,vec_X,vec_p)  # now a 10000 x 2 array

out, out_m, out_s, out_x, out_M, out_S, out_X = f(vec_m, vec_s,vec_x, vec_m, vec_S,vec_X,vec_p)
```


The vectorized evaluation is optimized so that it is much faster to make a vectorized call rather than iterate on each point. 

!!! note
    In the preceding example, the parameters are constant for all evaluations, yet they are repeated. This is not mandatory, and the call ``f(vec_m, vec_s, vec_x, vec_M, vec_S, vec_X, p)`` should work exactly as if `p` had been repeated along the first axis. We follow there numba's ``guvectorize`` conventions, even though they slightly differ from numpy's ones.


## Exogenous shock

The `exogenous` field contains information about the driving process. To get its default, discretized version, one can call `model.exogenous.discretize()`.


## Options structure


The ``model.options`` structure holds an information required by a particular solution method. For instance, for global methods, ``model.options['grid']`` is supposed to hold the boundaries and the number nodes at which to interpolate.

```python
display( model.options['grid'] )
```

```

File: docs/simulation.md
```md
Simulation
==========

![mkapi](dolo.algos.simulations.simulate)
```

File: docs/steady_state.md
```md
Steady-state
============

The deterministic state of a model corresponds to steady-state values
$\overline{m}$ of the exogenous process. States and controls satisfy:

> $\overline{s} = g\left(\overline{m}, \overline{s}, \overline{x}, \overline{m} \right)$
>
> $0 = \left[ f\left(\overline{m}, \overline{s}, \overline{x}, \overline{m}, \overline{s}, \overline{x} \right) \right]$

where $g$ is the state transition function, and $f$ is the arbitrage
equation. Note that the shocks, $\epsilon$, are held at their
deterministic mean.

The steady state function consists in solving the system of arbitrage
equations for the steady state values of the controls, $\overline{x}$,
which can then be used along with the transition function to find the
steady state values of the state variables, $\overline{s}$.

![mkapi](dolo.algos.steady_state.residuals)

```

File: README.md
```md
Complete documentation with installation instruction, available at https://www.econforge.org/dolo.py.

Join the chat at https://gitter.im/EconForge/dolo

[![codecov](https://codecov.io/gh/EconForge/dolo.py/branch/master/graph/badge.svg?token=hLAd1OaTRp)](https://codecov.io/gh/EconForge/dolo.py)

![CI](https://github.com/EconForge/dolo.py/workflows/CI/badge.svg)

![Publish docs via GitHub Pages](https://github.com/EconForge/dolo.py/workflows/Publish%20docs%20via%20GitHub%20Pages/badge.svg)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EconForge/dolo.git/master?urlpath=lab)

```

File: examples/models_/rbc_iid_ar1.yaml
```yaml
name: Real Business Cycle

model_type: dtcc

symbols:

   exogenous: [e_z]
   states: [z, k]
   controls: [n, i]
   expectations: [m]
   values: [V]
   parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]
   rewards: [u]

definitions:
    y: exp(z)*k^alpha*n^(1-alpha)
    c: y - i
    rk: alpha*y/k
    w: (1-alpha)*y/n

equations:

    arbitrage:
        - chi*n^eta*c^sigma - w                      | 0.1 <= n <= 1.0
        - 1 - beta*(c/c(1))^(sigma)*(1-delta+rk(1))  | 0.0 <= i <= 1.0


    transition:
        - z = rho*z(-1) + e_z
        - k = (1-delta)*k(-1) + i(-1)

    value:
        - V = c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta) + beta*V(1)

    felicity:
        - u =  c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)

    expectation:
        - m = beta/c(1)^sigma*(1-delta+rk(1))

    direct_response:
        - n = ((1-alpha)*exp(z)*k^alpha*m/chi)^(1/(eta+alpha))
        - i = exp(z)*k^alpha*n^(1-alpha) - (m)^(-1/sigma)

calibration:

    # parameters
    beta : 0.99
    phi: 1
    delta : 0.025
    alpha : 0.33
    rho : 0.8
    sigma: 5
    eta: 1
    sig_z: 0.016
    zbar: 0
    chi : w/c^sigma/n^eta
    c_i: 1.5
    c_y: 0.5
    e_z: 0.0

    # endogenous variables
    n: 0.33
    z: zbar
    rk: 1/beta-1+delta
    w: (1-alpha)*exp(z)*(k/n)^(alpha)
    k: n/(rk/alpha)^(1/(1-alpha))
    y: exp(z)*k^alpha*n^(1-alpha)
    i: delta*k
    c: y - i
    V: log(c)/(1-beta)
    u: c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)
    m: beta/c^sigma*(1-delta+rk)

exogenous: !Normal
    Sigma: [[sig_z**2]]

domain:
    z: [-sig_z, sig_z]
    k: [-k*0.5, k*1.5]

options:
    grid: !Cartesian
        orders: [20, 50]

```

File: docs/installation.md
```md
## Basic installation

Dolo can be installed in several ways:

- with anaconda (recommended):

  `conda install -c conda-forge dolo`

- with pip

  `pip install dolo`


## Developper's installation

Dolo uses `poetry` as package manager, so you probably need to install poetry before you start developing the package.

```

File: examples/models_/rbc2.yaml
```yaml
name: Real Business Cycle

symbols:

   exogenous: [e_z]
   states: [z, k]
   controls: [n, i]
   expectations: [m]
   values: [V]
   parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]
   rewards: [u]

definitions:
    y: exp(z)*k^alpha*n^(1-alpha)
    c: y - i
    rk: alpha*y/k
    w: (1-alpha)*y/n

equations:

    arbitrage:
        - chi*n^eta*c^sigma - w                     | 0.0 <= n <= inf
        - 1 - beta*(c/c(1))^(sigma)*(1-delta+rk(1))  | 0.0 <= i <= inf


    transition:
        - z = rho*z(-1) + e_z
        - k = (1-delta)*k(-1) + i(-1)

    value:
        - V = c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta) + beta*V(1)

    felicity:
        - u =  c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)

    expectation:
        - m = beta/c(1)^sigma*(1-delta+rk(1))

    direct_response:
        - n = ((1-alpha)*exp(z)*k^alpha*m/chi)^(1/(eta+alpha))
        - i = exp(z)*k^alpha*n^(1-alpha) - (m)^(-1/sigma)

calibration:

    # parameters
    beta : 0.99
    phi: 1
    delta : 0.025
    alpha : 0.33
    rho : 0.8
    sigma: 5
    eta: 1
    sig_z: 0.016
    zbar: 0
    chi : w/c^sigma/n^eta
    c_i: 1.5
    c_y: 0.5
    e_z: 0.0

    # endogenous variables
    n: 0.33
    z: zbar
    rk: 1/beta-1+delta
    w: (1-alpha)*exp(z)*(k/n)^(alpha)
    k: n/(rk/alpha)^(1/(1-alpha))
    y: exp(z)*k^alpha*n^(1-alpha)
    i: delta*k
    c: y - i
    V: log(c)/(1-beta)
    u: c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)
    m: beta/c^sigma*(1-delta+rk)

exogenous: !Normal
    Sigma: [[sig_z**2]]

domain:
    z: [-2*sig_z/(1-rho^2)^0.5,  2*sig_z/(1-rho^2)^0.5]
    k: [ k*0.5, k*1.5]

options:
    grid: !Cartesian
        n: [20, 20]
# options:
#     grid: !Smolyak
#         mu: 3
#         # orders: [5, 50]

```

File: examples/models_/rbc_mc.yaml
```yaml
name: Real Business Cycle

symbols:

   exogenous: [z,p]
   states: [k]
   controls: [n, i]
   expectations: [m]
   values: [V]
   parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]
   rewards: [u]

definitions:
    y: exp(z)*k^alpha*n^(1-alpha)
    c: y - i
    rk: alpha*y/k
    w: (1-alpha)*y/n

equations:

    arbitrage:
        - chi*n^eta*c^sigma - w(1)                   | 0.01 <= n <= 1.0
        - 1 - beta*(c/c(1))^(sigma)*(1-delta+rk(1))  | 0.00 <= i <= 1.0

    transition:
        - k = (1-delta)*k(-1) + i(-1)

    value:
        - V = c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta) + beta*V(1)

    felicity:
        - u =  c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)

    expectation:
        - m = beta/c(1)^sigma*(1-delta+rk(1))

    direct_response:
        - n = ((1-alpha)*exp(z)*k^alpha*m/chi)^(1/(eta+alpha))
        - i = exp(z)*k^alpha*n^(1-alpha) - (m)^(-1/sigma)

calibration:

    # parameters
    beta: 0.99
    phi: 1
    delta : 0.025
    alpha : 0.33
    rho : 0.8
    sigma: 5
    eta: 1
    sig_z: 0.016
    zbar: 0
    chi : w/c^sigma/n^eta
    c_i: 1.5
    c_y: 0.5
    e_z: 0.0
    m: 0
    V0: (c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta))/(1-beta)

    # endogenous variables
    n: 0.33
    z: zbar
    rk: 1/beta-1+delta
    w: (1-alpha)*exp(z)*(k/n)^(alpha)
    k: n/(rk/alpha)^(1/(1-alpha))
    y: exp(z)*k^alpha*n^(1-alpha)
    i: delta*k
    c: y - i
    V: log(c)/(1-beta)
    u: c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)


exogenous: !MarkovChain
    values: [[-0.01, 0.1],[0.01, 0.1]]
    transitions: [[0.9, 0.1], [0.1, 0.9]]

domain:
    k: [k*0.2, k*2.0]

options:
    grid: !Cartesian
        orders: [50]

```

File: examples/models_/rbc_product.yaml
```yaml
name: Real Business Cycle

symbols:

   exogenous: [e_z, e_p]
   states: [z, k]
   controls: [n, i]
   expectations: [m]
   values: [V]
   parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]
   rewards: [u]

definitions:
    y: exp(z)*k^alpha*n^(1-alpha)
    c: y - i
    rk: alpha*y/k
    w: (1-alpha)*y/n

equations:

    arbitrage:
        - chi*n^eta*c^sigma - w                     | 0.0 <= n <= inf
        - 1 - beta*(c/c(1))^(sigma)*(1-delta+rk(1))  | 0.0 <= i <= inf


    transition:
        - z = rho*z(-1) + e_z
        - k = (1-delta)*k(-1) + i(-1)

    value:
        - V = c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta) + beta*V(1)

    felicity:
        - u =  c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)

    expectation:
        - m = beta/c(1)^sigma*(1-delta+rk(1))

    direct_response:
        - n = ((1-alpha)*exp(z)*k^alpha*m/chi)^(1/(eta+alpha))
        - i = exp(z)*k^alpha*n^(1-alpha) - (m)^(-1/sigma)

calibration:

    # parameters
    beta : 0.99
    phi: 1
    delta : 0.025
    alpha : 0.33
    rho : 0.8
    sigma: 5
    eta: 1
    sig_z: 0.016
    zbar: 0
    chi : w/c^sigma/n^eta
    c_i: 1.5
    c_y: 0.5
    e_z: 0.0

    # endogenous variables
    n: 0.33
    z: zbar
    rk: 1/beta-1+delta
    w: (1-alpha)*exp(z)*(k/n)^(alpha)
    k: n/(rk/alpha)^(1/(1-alpha))
    y: exp(z)*k^alpha*n^(1-alpha)
    i: delta*k
    c: y - i
    V: log(c)/(1-beta)
    u: c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)
    m: beta/c^sigma*(1-delta+rk)

# exogenous: !Product
#     - !UNormal
#         σ: 0.01
#     - !UNormal
#         σ: 0.02

# exogenous: !Product
#     - !UNormal
#         σ: 0.01
#     - !UNormal
#         σ: 0.02


domain:
    z: [-2*sig_z/(1-rho^2)^0.5,  2*sig_z/(1-rho^2)^0.5]
    k: [ k*0.5, k*1.5]

options:
    grid: !Cartesian
        n: [20, 20]
# options:
#     grid: !Smolyak
#         mu: 3
#         # orders: [5, 50]

```

File: examples/models_/rbc_product_newstyle.yaml
```yaml
name: Real Business Cycle

symbols:

   exogenous: [e_z, e_p, e_r]
   states: [z, k]
   controls: [n, i]
   expectations: [m]
   values: [V]
   parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]
   rewards: [u]

definitions:
    y: exp(z)*k^alpha*n^(1-alpha)
    c: y - i
    rk: alpha*y/k
    w: (1-alpha)*y/n

equations:

    arbitrage:
        - chi*n^eta*c^sigma - w                     | 0.0 <= n <= inf
        - 1 - beta*(c/c(1))^(sigma)*(1-delta+rk(1))  | 0.0 <= i <= inf


    transition:
        - z = rho*z(-1) + e_z
        - k = (1-delta)*k(-1) + i(-1)

    value:
        - V = c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta) + beta*V(1)

    felicity:
        - u =  c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)

    expectation:
        - m = beta/c(1)^sigma*(1-delta+rk(1))

    direct_response:
        - n = ((1-alpha)*exp(z)*k^alpha*m/chi)^(1/(eta+alpha))
        - i = exp(z)*k^alpha*n^(1-alpha) - (m)^(-1/sigma)

calibration:

    # parameters
    beta : 0.99
    phi: 1
    delta : 0.025
    alpha : 0.33
    rho : 0.8
    sigma: 5
    eta: 1
    sig_z: 0.016
    zbar: 0
    chi : w/c^sigma/n^eta
    c_i: 1.5
    c_y: 0.5
    e_z: 0.0

    # endogenous variables
    n: 0.33
    z: zbar
    rk: 1/beta-1+delta
    w: (1-alpha)*exp(z)*(k/n)^(alpha)
    k: n/(rk/alpha)^(1/(1-alpha))
    y: exp(z)*k^alpha*n^(1-alpha)
    i: delta*k
    c: y - i
    V: log(c)/(1-beta)
    u: c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)
    m: beta/c^sigma*(1-delta+rk)

exogenous:
    e_z: !UNormal
        σ: 0.01
    e_p, e_r: !Normal
        Σ: [[0.1, 0.02], [0.02, 0.4]]

domain:
    z: [-2*sig_z/(1-rho^2)^0.5,  2*sig_z/(1-rho^2)^0.5]
    k: [ k*0.5, k*1.5]

options:
    grid: !Cartesian
        n: [20, 20]
# options:
#     grid: !Smolyak
#         mu: 3
#         # orders: [5, 50]

```

File: docs/parameterized_expectations.md
```md
Parameterized expectations
==========================

We consider an fgh model, that is a model with the form:

> $s_t = g\left(s_{t-1}, x_{t-1}, \epsilon_t \right)$
>
> $0 = f\left(s_{t}, x_{t}, E_t[h(s_{t+1}, x_{t+1})] \right)$

where $g$ is the state transition function, $f$ is the arbitrage
equation, and $h$ is the expectations function (more accurately, it is
the function over which expectations are taken).

The parameterized expectations algorithm consists in approximating the
expectations function, $h$, and solving for the associated optimal
controls, $x_t = x(s_t)$.

At step $n$, with a current guess of the control, $x(s_t) = \varphi^n(s_t)$, and expectations function, $h(s_t,x_t) = \psi^n(s_t)$ :

:   -   Compute the conditional expectation $z_t = E_t[\varphi^n(s_t)]$
    -   Given the expectation, update the optimal control by solving
        $0 = \left( f\left(s_{t}, \varphi^{n+1}(s), z_t \right) \right)$
    -   Given the optimal control, update the expectations function
        $\psi^{n+1}(s_t) = h(s_t, \varphi^{n+1}(s_t))$

TODO: link to updated function.
```

File: examples/models_/rbc.yaml
```yaml
name: Real Business Cycle

symbols:

   exogenous: [e_z]
   states: [z, k]
   controls: [n, i]
   expectations: [m]
   values: [V]
   parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]
   rewards: [u]

definitions:
    y: exp(z)*k^alpha*n^(1-alpha)
    c: y - i
    rk: alpha*y/k
    w: (1-alpha)*y/n

equations:

    arbitrage:
        - chi*n^eta*c^sigma - w                     | 0.0 <= n <= inf
        - 1 - beta*(c/c(1))^(sigma)*(1-delta+rk(1))  | 0.0 <= i <= inf


    transition:
        - z = rho*z(-1) + e_z
        - k = (1-delta)*k(-1) + i(-1)

    value:
        - V = c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta) + beta*V(1)

    felicity:
        - u =  c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)

    expectation:
        - m = beta/c(1)^sigma*(1-delta+rk(1))

    direct_response:
        - n = ((1-alpha)*exp(z)*k^alpha*m/chi)^(1/(eta+alpha))
        - i = exp(z)*k^alpha*n^(1-alpha) - (m)^(-1/sigma)

calibration:

    # parameters
    beta : 0.99
    phi: 1
    delta : 0.025
    alpha : 0.33
    rho : 0.8
    sigma: 5
    eta: 1
    sig_z: 0.016
    zbar: 0
    chi : w/c^sigma/n^eta
    c_i: 1.5
    c_y: 0.5
    e_z: 0.0
    n: 0.33
    z: zbar
    rk: 1/beta-1+delta
    w: (1-alpha)*exp(z)*(k/n)^(alpha)
    k: n/(rk/alpha)^(1/(1-alpha))
    y: exp(z)*k^alpha*n^(1-alpha)
    i: delta*k
    c: y - i
    V: log(c)/(1-beta)
    u: c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)
    m: beta/c^sigma*(1-delta+rk)
    kss: 10

exogenous: !UNormal
    sigma: 0.01

domain:
    z: [-2*sig_z/(1-rho^2)^0.5,  2*sig_z/(1-rho^2)^0.5]
    k: [ kss*0.5, kss*1.5]

options:
    grid: !Cartesian
        n: [20, 20]
    discrete_choices: [n]
# options:
#     grid: !Smolyak
#         mu: 3
#         # orders: [5, 50]

```

File: examples/models_/rbc_iid.yaml
```yaml
name: Real Business Cycle

symbols:

   exogenous: [e_z]
   states: [z, k]
   controls: [n, i]
   expectations: [m]
   values: [V]
   parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]
   rewards: [u]

definitions:
    y: exp(z)*k^alpha*n^(1-alpha)
    c: y - i
    rk: alpha*y/k
    w: (1-alpha)*y/n

equations:

    arbitrage:
        - chi*n^eta*c^sigma - w                     | 0.0 <= n <= inf
        - 1 - beta*(c/c(1))^(sigma)*(1-delta+rk(1))  | 0.0 <= i <= inf


    transition:
        - z = rho*z(-1) + e_z
        - k = (1-delta)*k(-1) + i(-1)

    value:
        - V = c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta) + beta*V(1)

    felicity:
        - u =  c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)

    expectation:
        - m = beta/c(1)^sigma*(1-delta+rk(1))

    direct_response:
        - n = ((1-alpha)*exp(z)*k^alpha*m/chi)^(1/(eta+alpha))
        - i = exp(z)*k^alpha*n^(1-alpha) - (m)^(-1/sigma)

calibration:

    # parameters
    beta : 0.99
    phi: 1
    delta : 0.025
    alpha : 0.33
    rho : 0.8
    sigma: 5
    eta: 1
    sig_z: 0.016
    zbar: 0
    chi : w/c^sigma/n^eta
    c_i: 1.5
    c_y: 0.5
    e_z: 0.0

    # endogenous variables
    n: 0.33
    z: zbar
    rk: 1/beta-1+delta
    w: (1-alpha)*exp(z)*(k/n)^(alpha)
    k: n/(rk/alpha)^(1/(1-alpha))
    y: exp(z)*k^alpha*n^(1-alpha)
    i: delta*k
    c: y - i
    V: log(c)/(1-beta)
    u: c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)
    m: beta/c^sigma*(1-delta+rk)

exogenous: !Normal
    Σ: [[sig_z**2]]

domain:
    z: [-2*sig_z/(1-rho^2)^0.5, 2*sig_z/(1-rho^2)^0.5]
    k: [k*0.5, k*1.5]

options:
    grid: !Cartesian
        orders: [5, 50]

```

File: examples/models_/rbc_mixture.yaml
```yaml
name: Real Business Cycle

symbols:

   exogenous: [e_z]
   states: [z, k]
   controls: [n, i]
   expectations: [m]
   values: [V]
   parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]
   rewards: [u]

definitions:
    y: exp(z)*k^alpha*n^(1-alpha)
    c: y - i
    rk: alpha*y/k
    w: (1-alpha)*y/n

equations:

    arbitrage:
        - chi*n^eta*c^sigma - w                     | 0.0 <= n <= inf
        - 1 - beta*(c/c(1))^(sigma)*(1-delta+rk(1))  | 0.0 <= i <= inf


    transition:
        - z = rho*z(-1) + e_z
        - k = (1-delta)*k(-1) + i(-1)

    value:
        - V = c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta) + beta*V(1)

    felicity:
        - u =  c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)

    expectation:
        - m = beta/c(1)^sigma*(1-delta+rk(1))

    direct_response:
        - n = ((1-alpha)*exp(z)*k^alpha*m/chi)^(1/(eta+alpha))
        - i = exp(z)*k^alpha*n^(1-alpha) - (m)^(-1/sigma)

calibration:

    # parameters
    beta : 0.99
    phi: 1
    delta : 0.025
    alpha : 0.33
    rho : 0.8
    sigma: 5
    eta: 1
    sig_z: 0.016
    zbar: 0
    chi : w/c^sigma/n^eta
    c_i: 1.5
    c_y: 0.5
    e_z: 0.0

    # endogenous variables
    n: 0.33
    z: zbar
    rk: 1/beta-1+delta
    w: (1-alpha)*exp(z)*(k/n)^(alpha)
    k: n/(rk/alpha)^(1/(1-alpha))
    y: exp(z)*k^alpha*n^(1-alpha)
    i: delta*k
    c: y - i
    V: log(c)/(1-beta)
    u: c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)
    m: beta/c^sigma*(1-delta+rk)
#
# exogenous: !UNormal
#     μ: 0.0
#     σ: 0.01
#

exogenous: !Mixture
    index: !Bernouilli
        π: 0.4
    distributions:
        0: !UNormal
            μ: 0.0
            σ: 0.01
        1: !UNormal
            σ: 0.05
            μ: 0.8

domain:
    z: [-2*sig_z/(1-rho^2)^0.5,  2*sig_z/(1-rho^2)^0.5]
    k: [ k*0.5, k*1.5]

options:
    grid: !Cartesian
        n: [20, 20]
# options:
#     grid: !Smolyak
#         mu: 3
#         # orders: [5, 50]

```

File: examples/models_/rbc_condition.yaml
```yaml
name: Real Business Cycle

symbols:

   exogenous: [e_z, e_p]
   states: [z, k]
   controls: [n, i]
   expectations: [m]
   values: [V]
   parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]
   rewards: [u]

definitions:
    y: exp(z)*k^alpha*n^(1-alpha)
    c: y - i
    rk: alpha*y/k
    w: (1-alpha)*y/n

equations:

    arbitrage:
        - chi*n^eta*c^sigma - w                     | 0.0 <= n <= inf
        - 1 - beta*(c/c(1))^(sigma)*(1-delta+rk(1))  | 0.0 <= i <= inf


    transition:
        - z = rho*z(-1) + e_z
        - k = (1-delta)*k(-1) + i(-1)

    value:
        - V = c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta) + beta*V(1)

    felicity:
        - u =  c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)

    expectation:
        - m = beta/c(1)^sigma*(1-delta+rk(1))

    direct_response:
        - n = ((1-alpha)*exp(z)*k^alpha*m/chi)^(1/(eta+alpha))
        - i = exp(z)*k^alpha*n^(1-alpha) - (m)^(-1/sigma)

calibration:

    # parameters
    beta : 0.99
    phi: 1
    delta : 0.025
    alpha : 0.33
    rho : 0.8
    sigma: 5
    eta: 1
    sig_z: 0.016
    zbar: 0
    chi : w/c^sigma/n^eta
    c_i: 1.5
    c_y: 0.5
    e_z: 0.0

    # endogenous variables
    n: 0.33
    z: zbar
    rk: 1/beta-1+delta
    w: (1-alpha)*exp(z)*(k/n)^(alpha)
    k: n/(rk/alpha)^(1/(1-alpha))
    y: exp(z)*k^alpha*n^(1-alpha)
    i: delta*k
    c: y - i
    V: log(c)/(1-beta)
    u: c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)
    m: beta/c^sigma*(1-delta+rk)

# exogenous: !Product
#     - !UNormal
#         σ: 0.01
#     - !UNormal
#         σ: 0.02

# exogenous: !Product
#     - !UNormal
#         σ: 0.01
#     - !UNormal
#         σ: 0.02
exogenous: !Conditional
    condition: !UNormal
        mu: 0.0
        sigma: 0.2
    type: Markov
    arguments: !Function
        arguments: [x]
        value:
          states: [0.1, 0.2]
          transitions: !Matrix
              [[1-0.1-x, 0.1+x],
               [0.5,       0.5]]

      # ])

domain:
    z: [-2*sig_z/(1-rho^2)^0.5,  2*sig_z/(1-rho^2)^0.5]
    k: [ k*0.5, k*1.5]

options:
    grid: !Cartesian
        n: [20, 20]
# options:
#     grid: !Smolyak
#         mu: 3
#         # orders: [5, 50]

```

File: examples/models_/rbc_taxes.yaml
```yaml
name: taxes

symbols:
   states:  [z, k, g]
   controls: [i, n]
   exogenous: [e_g]
   parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar ]

definitions:
    rk: alpha*z*(n/k)^(1-alpha)
    w: (1-alpha)*z*(k/n)^(alpha)
    y: z*k^alpha*n^(1-alpha)
    c: k*rk + w*n - i - g

equations:

   arbitrage:
      - 1 - beta*(c/c(1))^(sigma)*(1-delta+rk(1))    | 0 <= i <= inf
      - chi*n^eta*c^sigma - w                        | 0 <= n <= inf

   transition:
      - z = (1-rho)*zbar + rho*z(-1)
      - k = (1-delta)*k(-1) + i(-1)
      - g = e_g


calibration:

      beta : 0.99
      phi: 1
      chi : w/c^sigma/n^eta
      delta : 0.025
      alpha : 0.33
      rho : 0.8
      sigma: 1
      eta: 1
      zbar: 1

      z: zbar
      rk: 1/beta-1+delta
      w: (1-alpha)*z*(k/n)^(alpha)
      n: 0.33
      k: n/(rk/alpha)^(1/(1-alpha))
      i: delta*k
      c: z*k^alpha*n^(1-alpha) - i
      y: z*k^alpha*n^(1-alpha)
      g: 0


options:
    exogenous: !Normal
        Sigma: [ [ 0.0015 ] ]

```

File: examples/models_/capital.yaml
```yaml
name: Neoclassical model of capital accumulation

symbols:

    states: [k, A]
    controls: [i]
    exogenous: [epsilon]
    parameters: [beta, gamma, delta, theta, rho]

definitions:

    c: A*k^theta - i
    r_I: A*theta*k^(theta-1) + (1-delta)

equations:

    transition:
        - k = (1-delta)*k(-1) + i(-1)
        - A = 1 + epsilon + rho*A(-1)

    arbitrage:
        - 1 - beta*( (c(1)/c)^(-gamma)*r_I(1) )

############################
calibration:

    ## steady state

    # controls
    i: ( (1/beta - (1-delta))/theta )^(1/(theta-1)) * delta

    # states
    A: 1
    k: i/delta

    # auxiliary
    c: A*k^theta - i
    r_I: 1/beta

    # parameters:
    beta: 0.96
    gamma: 4.0
    delta: 0.1
    theta: 0.3
    rho: 0.0

exogenous: !Normal
    Sigma: [[ 0.00001 ]]

```

File: examples/models_/NK_dsge.yaml
```yaml
name: New Keynesian DSGE

symbols:

  states: [Delta__L, a, cp]
  controls: [c, n, F1, F2]
  expectations: [m1]
  exogenous: [u__a, u__tau]
  parameters: [beta, psi, epsilon, phi, theta, tau, chi, sig_a, rho__a, rho__tau, sig_tau]

definitions:

      pi: ((1-(F1/F2)^(1-epsilon)*(1-theta))/theta)^(1/(epsilon-1))
      r: (1/beta)*(pi)^phi
      rw: chi*(n)^psi*(c)
      mc: (1-tau*exp(cp))*(rw)/(exp(a))
      Delta: theta*(pi)^epsilon*Delta__L+(1-theta)*((F1)/(F2))^(-epsilon)

equations:

  arbitrage:
    - 1/c-beta*r/(c(1)*pi(1))
    - (epsilon/(epsilon-1))*mc + theta*beta*pi(1)^epsilon*F1(1)-F1
    - 1+theta*beta*pi(1)^(epsilon-1)*F2(1)-F2
    - c-exp(a)/Delta*n

  transition:
    - Delta__L=Delta(-1)
    - a=rho__a*a(-1)+u__a
    - cp=rho__tau*cp(-1)+u__tau

calibration:

  # parameters calibration
  beta: .995
  psi: 1 #1.0
  epsilon: 11
  phi: 1.5
  theta: .4
  tau: 1/epsilon
  chi: 1.0
  sig_a: .0025
  rho__a: .9
  sig_tau: .025
  rho__tau: 0


  # variable steady states / initial conditions
  pi: 1
  r: 1/beta
  Delta: 1
  Delta__L: Delta
  n: 1
  c: 1
  mc: (1-tau)*(rw)/(exp(a))
  y: 1
  F1:  (epsilon/(epsilon-1))*mc/(1-theta*beta*((pi))^epsilon)
  F2:  1/(1-theta*beta*((pi)^(epsilon-1)))
  a: 0
  u__a: 0
  cp: 0
  u__tau: 0

exogenous: !Normal
  Sigma: [[sig_a^2, 0], [0, sig_tau^2]]

domain:
    Delta__L: [1.0, 1.05]
    a: [-3*((sig_a^2)/(1-rho__a^2))^.5, 3*((sig_a^2)/(1-rho__a^2))^.5]
    cp: [-3*sig_tau, 3*sig_tau]

options:
  grid: !CartesianGrid
    orders: [20, 20, 20]

```

File: examples/models_/sudden_stop.yaml
```yaml
# This file adapts the model described in
# "From Sudden Stops to Fisherian Deflation, Quantitative Theory and Policy"
# by Anton Korinek and Enrique G. Mendoza

name: Sudden Stop (General)

symbols:

    exogenous: [y]
    states: [l]
    controls: [b, lam]
    values: [V, Vc]
    parameters: [beta, R, sigma, a, mu, kappa, delta_y, pi, lam_inf]


definitions:
    c: 1 + y + l*R - b

equations:

    transition:
        - l = b(-1)

    arbitrage:
        - lam = b/c
        - 1 - beta*(c(1)/c)^(-sigma)*R    |  lam_inf <= lam <= inf

    value:
        - V = c^(1.0-sigma)/(1.0-sigma) + beta*V(1)
        - Vc = c^(1.0-sigma)/(1.0-sigma)

calibration:

    beta: 0.95
    R: 1.03
    sigma: 2.0
    a: 1/3
    mu: 0.8
    kappa: 1.3
    delta_y: 0.03
    pi: 0.05
    lam_inf: -0.2
    y: 1.0
    c: 1.0 + y
    b: 0.0
    l: 0.0
    lam: 0.0

    V: c^(1.0-sigma)/(1.0-sigma)/(1.0-beta)
    Vc: c^(1.0-sigma)/(1.0-sigma)


exogenous: !MarkovChain
    values: [[ 1.0-delta_y ],  # bad state
             [ 1.0 ]]          # good state
    transitions: [[ 0.5, 1-0.5 ],   # probabilities   [p(L|L), p(H|L)]
                  [ 0.5, 0.5 ]]     # probabilities   [p(L|H), p(H|H)]

domain:
    l: [-1, 1]

options:
    grid: !Cartesian
        orders: [10]

```

File: examples/models_/Figv4_1191.yaml
```yaml
#  Adapted from the Dynare .mod file Figv3_1161.mod for RMT3
name: Figv3_1161

symbols:
    states: [k]
    controls: [c]
    exogenous: [g]
    parameters: [beta, gamma, delta, alpha, A, tau_c, tau_k]

definitions:
    eta: alpha * A * k ** (alpha - 1)
    w:   A * k ** alpha - k * eta
    R:  ((1. - tau_k) * (alpha * A * k ** (alpha - 1) - delta) + 1.)

equations:

    arbitrage:
        # Equation 11.6.3
        # - c^(-gamma)= beta*(c(+1)^(-gamma))*((1+tau_c)/(1+tau_c(+1)))*((1-delta) + (1-tau_k(+1))*alpha*A*k^(alpha-1))
        # - c^(-gamma)= beta*(c(+1)^(-gamma))*(1-delta + alpha*A*k^(alpha-1))
        - c^(-gamma) = beta*(c(+1)^(-gamma))*((1+tau_c)/(1+tau_c))*((1-delta) + (1-tau_k)*alpha*A*k^(alpha-1))

    transition:
        # Equation 11.6.1
        - k = A*k(-1)^alpha+(1-delta)*k(-1)-c(-1)-g


calibration:

    beta  : .95
    gamma : 2.0
    delta : 0.2
    alpha : .33
    A : 1.
    tau_c: 0.0
    tau_k: 0.0
    g: 0.2

    k: ((1/beta - 1 + delta)/alpha)^(1/(alpha-1))
    c: k^alpha - delta*k - g
    eta: alpha * A * k ** (alpha - 1)
    w: A * k ** alpha - k * eta
    R: eta-delta +1


exogenous: !Normal

    Sigma: [ [ 0.00001] ]

```

File: examples/models_/open_economy.yaml
```yaml
name: Open economy

description: Two endowment economies with one riskless bond

symbols:

  states: [W_1, W_2]
  controls: [p_f, db_f]
  exogenous: [epsilon_1, epsilon_2]
  parameters: [beta, gamma, ybar_1, ybar_2, kappa, dumb]

definitions:
  c_1: W_1 - db_f*p_f
  c_2: W_2 + db_f*p_f

equations:

    transition:
        - W_1 = ybar_1 + epsilon_1 + dumb*W_1(-1) + db_f(-1)
        - W_2 = ybar_2 + epsilon_2 + dumb*W_2(-1) - db_f(-1)

    arbitrage:
        - beta*( c_1^(-kappa)*(c_1(1)/c_1)^(-gamma) + c_2^(-kappa)*(c_2(1)/c_2)^(-gamma) )/2 - p_f
        - beta*( c_1^(-kappa)*(c_1(1)/c_1)^(-gamma) - c_2^(-kappa)*(c_2(1)/c_2)^(-gamma) )    |  -inf <= db_f <= inf

############################
calibration:
    # steady_state
    p_f: beta
    db_f: 0
    W_1: 1
    W_2: 1
    c_1: W_1
    c_2: W_2


    # parameters:
    beta: 0.96
    gamma: 4.0
    ybar_1: 1.0
    ybar_2: 1.0
    kappa: 0.00
    dumb: 0

exogenous: !Normal
    Sigma: [[ 0.005, 0 ],
            [ 0, 0.005 ]]
domain:
    W_1: [0.7, 1.3]
    W_2: [0.7, 1.3]

options:
    grid: !Cartesian
        orders: [5, 5]

```

File: examples/models_/rbc_ar1.yaml
```yaml
name: Real Business Cycle

symbols:

   exogenous: [z]
   states: [k]
   controls: [n, i]
   expectations: [m]
   values: [V]
   parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]
   rewards: [u]

definitions:
    y: exp(z)*k^alpha*n^(1-alpha)
    c: y - i
    rk: alpha*y/k
    w: (1-alpha)*y/n

equations:

    arbitrage:
        - chi*n^eta*c^sigma - w                      | 0 <= n <= inf
        - 1 - beta*(c/c(1))^(sigma)*(1-delta+rk(1))  | 0 <= i <= inf

    transition:
        - k = (1-delta)*k(-1) + i(-1)

    value:
        - V = c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta) + beta*V(1)

    felicity:
        - u =  c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)

    expectation:
        - m = beta/c(1)^sigma*(1-delta+rk(1))

    direct_response:
        - n = ((1-alpha)*exp(z)*k^alpha*m/chi)^(1/(eta+alpha))
        - i = exp(z)*k^alpha*n^(1-alpha) - (m)^(-1/sigma)

calibration:

    # parameters
    beta: 0.99
    phi: 1
    delta : 0.025
    alpha : 0.33
    rho : 0.8
    sigma: 5
    eta: 1
    sig_z: 0.016
    zbar: 0
    chi : w/c^sigma/n^eta
    c_i: 1.5
    c_y: 0.5
    e_z: 0.0
    m: 0
    V0: (c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta))/(1-beta)

    # endogenous variables
    n: 0.33
    z: zbar
    rk: 1/beta-1+delta
    w: (1-alpha)*exp(z)*(k/n)^(alpha)
    k: n/(rk/alpha)^(1/(1-alpha))
    y: exp(z)*k^alpha*n^(1-alpha)
    i: delta*k
    c: y - i
    V: log(c)/(1-beta)
    u: c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)




exogenous: !VAR1
    rho: 0.8
    Sigma: [[sig_z^2]]

domain:
    k: [k*0.5, k*1.5]

options:
    grid: !Cartesian
        orders: [20]

```

File: examples/models/rbc.yaml
```yaml
name: Real Business Cycle

symbols:

   exogenous: [e_z]
   states: [z, k]
   controls: [n, i]
   parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]

definitions: |
    y[t] = exp(z[t])*k[t]^alpha*n[t]^(1-alpha)
    c[t] = y[t] - i[t]
    rk[t] = alpha*y[t]/k[t]
    w[t] = (1-alpha)*y[t]/n[t]

equations:

    arbitrage: |
        chi*n[t]^eta*c[t]^sigma - w[t]                     ⟂ 0.0 <= n[t] <= inf
        1 - beta*(c[t]/c[t+1])^(sigma)*(1-delta+rk[t+1])   ⟂ -inf <= i[t] <= inf

    transition: |
        z[t] = rho*z[t-1] + e_z
        k[t] = (1-delta)*k[t-1] + i[t-1]

calibration:

    # parameters
    beta : 0.99
    phi: 1
    delta : 0.025
    alpha : 0.33
    rho : 0.8
    sigma: 5
    eta: 1
    sig_z: 0.016
    zbar: 0
    chi : w/c^sigma/n^eta
    c_i: 1.5
    c_y: 0.5
    e_z: 0.0
    n: 0.33
    z: zbar
    rk: 1/beta-1+delta
    w: (1-alpha)*exp(z)*(k/n)^(alpha)
    k: n/(rk/alpha)^(1/(1-alpha))
    y: exp(z)*k^alpha*n^(1-alpha)
    i: delta*k
    c: y - i
    V: log(c)/(1-beta)
    u: c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)
    m: beta/c^sigma*(1-delta+rk)
    kss: 10

exogenous: !UNormal
    sigma: 0.01

domain:
    z: [-2*sig_z/(1-rho^2)^0.5,  2*sig_z/(1-rho^2)^0.5]
    k: [ kss*0.5, kss*1.5]

options:
    grid: !Cartesian
        n: [100, 100]
    discrete_choices: [n]

```

File: examples/models/rbc_mc.yaml
```yaml
name: Real Business Cycle

symbols:

   exogenous: [z,p]
   states: [k]
   controls: [n, i]
   expectations: [m]
   values: [V]
   parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]
   rewards: [u]

definitions:
    y[t]: exp(z[t])*k[t]^alpha*n[t]^(1-alpha)
    c[t]: y[t] - i[t]
    rk[t]: alpha*y[t]/k[t]
    w[t]: (1-alpha)*y[t]/n[t]

equations:

    arbitrage:
        - chi*n[t]^eta*c[t]^sigma - w[t]                   | 0.01 <= n[t] <= 1.0
        - 1 - beta*(c[t]/c[t+1])^(sigma)*(1-delta+rk[t+1])  | 0.00 <= i[t] <= 1.0

    transition:
        - k[t] = (1-delta)*k[t-1] + i[t-1]

    value:
        - V[t] = c[t]^(1-sigma)/(1-sigma) - chi*n[t]^(1+eta)/(1+eta) + beta*V[t+1]

    felicity:
        - u[t] =  c[t]^(1-sigma)/(1-sigma) - chi*n[t]^(1+eta)/(1+eta)

    expectation:
        - m[t] = beta/c[t+1]^sigma*(1-delta+rk[t+1])

    direct_response:
        - n[t] = ((1-alpha)*exp(z[t])*k[t]^alpha*m[t]/chi)^(1/(eta+alpha))
        - i[t] = exp(z[t])*k[t]^alpha*n[t]^(1-alpha) - (m[t])^(-1/sigma)

calibration:

    # parameters
    beta: 0.99
    phi: 1
    delta : 0.025
    alpha : 0.33
    rho : 0.8
    sigma: 5
    eta: 1
    sig_z: 0.016
    zbar: 0
    chi : w/c^sigma/n^eta
    c_i: 1.5
    c_y: 0.5
    e_z: 0.0
    m: 0
    V0: (c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta))/(1-beta)

    # endogenous variables
    n: 0.33
    z: zbar
    rk: 1/beta-1+delta
    w: (1-alpha)*exp(z)*(k/n)^(alpha)
    k: n/(rk/alpha)^(1/(1-alpha))
    y: exp(z)*k^alpha*n^(1-alpha)
    i: delta*k
    c: y - i
    V: log(c)/(1-beta)
    u: c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)


exogenous: !MarkovChain
    values: [[-0.01, 0.1],[0.01, 0.1]]
    transitions: [[0.9, 0.1], [0.1, 0.9]]

domain:
    k: [k*0.2, k*2.0]

options:
    grid: !Cartesian
        orders: [50]

```

File: examples/models/rbc_iid.yaml
```yaml
name: Real Business Cycle

symbols:

   exogenous: [e_z]
   states: [z, k]
   controls: [n, i]
   expectations: [m]
   values: [V]
   parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]
   rewards: [u]

definitions:
    y[t]: exp(z[t])*k[t]^alpha*n[t]^(1-alpha)
    c[t]: y[t] - i[t]
    rk[t]: alpha*y[t]/k[t]
    w[t]: (1-alpha)*y[t]/n[t]

equations:

    arbitrage:
        - chi*n[t]^eta*c[t]^sigma - w[t]                     ⟂ 0.0 <= n[t] <= inf
        - 1 - beta*(c[t]/c[t+1])^(sigma)*(1-delta+rk[t+1])   ⟂ 0.0 <= i[t] <= inf


    transition:
        - z[t] = rho*z[t-1] + e_z[t]
        - k[t] = (1-delta)*k[t-1] + i[t-1]

    value:
        - V[t] = c[t]^(1-sigma)/(1-sigma) - chi*n[t]^(1+eta)/(1+eta) + beta*V[t+1]

    felicity:
        - u[t] =  c[t]^(1-sigma)/(1-sigma) - chi*n[t]^(1+eta)/(1+eta)

    expectation:
        - m[t] = beta/c[t+1]^sigma*(1-delta+rk[t+1])

    direct_response:
        - n[t] = ((1-alpha)*exp(z[t])*k[t]^alpha*m[t]/chi)^(1/(eta+alpha))
        - i[t] = exp(z[t])*k[t]^alpha*n[t]^(1-alpha) - (m[t])^(-1/sigma)

calibration:

    # parameters
    beta : 0.99
    phi: 1
    delta : 0.025
    alpha : 0.33
    rho : 0.8
    sigma: 5
    eta: 1
    sig_z: 0.016
    zbar: 0
    chi : w/c^sigma/n^eta
    c_i: 1.5
    c_y: 0.5
    e_z: 0.0

    # endogenous variables
    n: 0.33
    z: zbar
    rk: 1/beta-1+delta
    w: (1-alpha)*exp(z)*(k/n)^(alpha)
    k: n/(rk/alpha)^(1/(1-alpha))
    y: exp(z)*k^alpha*n^(1-alpha)
    i: delta*k
    c: y - i
    V: log(c)/(1-beta)
    u: c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)
    m: beta/c^sigma*(1-delta+rk)

exogenous: !Normal
    Σ: [[sig_z**2]]

domain:
    z: [-2*sig_z/(1-rho^2)^0.5, 2*sig_z/(1-rho^2)^0.5]
    k: [k*0.5, k*1.5]

options:
    grid: !Cartesian
        orders: [5, 50]

```

File: examples/models/rmt3_ch11.yaml
```yaml
# Model in chapter 11 of Recursive Macroeconomic Theory 3rd edition by
# Ljvinquist and Sargent
name: fiscal_growth

symbols:
    states: [k, tau_c, tau_k]
    controls: [c]
    exogenous: [g, exog_tau_c, exog_tau_k]
    parameters: [beta, gamma, delta, alpha, A]

definitions:
    # Equations from 11.6.8
    eta[t]: alpha*A*k[t]^(alpha-1)
    w[t]: A*k[t]^alpha - k[t]*eta

equations:

    arbitrage:
        # Equation 11.6.3
        - beta*(c[t+1]/c[t])^(-gamma)*(1+tau_c[t])/(1+tau_c[t+1])*((1-tau_k[t+1])*(eta[t+1]-delta) + 1) - 1 | 0 <= c[t] <= inf

    transition:
        # Equation 11.6.1
        - k[t] = A*k[t-1]^alpha + (1-delta)*k[t-1]-c[t-1]-g[t]
        # We have the states tau_c and tau_k just follow exactly the sequence
        # of shocks that we supply.
        - tau_c[t] = exog_tau_c[t]
        - tau_k[t] = exog_tau_k[t]


calibration:
    # parameters
    beta: 0.95
    gamma: 2.0
    delta: 0.2
    alpha: 0.33
    A: 1.
    exog_tau_c: 0.0
    exog_tau_k: 0.0
    tau_c: exog_tau_c
    tau_k: exog_tau_k
    g: 0.2

    # steady_state
    k: ((1/beta - 1 + delta)/alpha)^(1/(alpha-1))
    c: k^alpha - delta*k - g

```

File: dolo/algos/egm.py
```py
import yaml
from dolo.numeric.decision_rule import DecisionRule
import numpy as np
from interpolation.splines import eval_linear
from dolo.compiler.model import Model
from .results import EGMResult


def egm(
    model: Model,
    dr0: DecisionRule = None,
    verbose: bool = False,
    details: bool = True,
    a_grid=None,
    η_tol=1e-6,
    maxit=1000,
    grid=None,
    dp=None,
):
    """
    a_grid: (numpy-array) vector of points used to discretize poststates; must be increasing
    """

    assert len(model.symbols["states"]) == 1
    assert (
        len(model.symbols["controls"]) == 1
    )  # we probably don't need this restriction

    from dolo.numeric.processes import IIDProcess

    iid_process = isinstance(model.exogenous, IIDProcess)

    def vprint(t):
        if verbose:
            print(t)

    p = model.calibration["parameters"]

    if grid is None and dp is None:
        grid, dp = model.discretize()

    s = grid["endo"].nodes

    funs = model.__original_gufunctions__
    h = funs["expectation"]
    gt = funs["half_transition"]
    τ = funs["direct_response_egm"]
    aτ = funs["reverse_state"]
    lb = funs["arbitrage_lb"]
    ub = funs["arbitrage_ub"]

    if dr0 is None:
        x0 = model.calibration["controls"]
        dr0 = lambda i, s: x0[None, :].repeat(s.shape[0], axis=0)

    n_m = dp.n_nodes
    n_x = len(model.symbols["controls"])

    if a_grid is None:
        raise Exception("You must supply a grid for the post-states.")

    assert a_grid.ndim == 1
    a = a_grid[:, None]
    N_a = a.shape[0]

    N = s.shape[0]

    n_h = len(model.symbols["expectations"])

    xa = np.zeros((n_m, N_a, n_x))
    sa = np.zeros((n_m, N_a, 1))
    xa0 = np.zeros((n_m, N_a, n_x))
    sa0 = np.zeros((n_m, N_a, 1))

    z = np.zeros((n_m, N_a, n_h))

    if verbose:
        headline = "|{0:^4} | {1:10} |".format("N", " Error")
        stars = "-" * len(headline)
        print(stars)
        print(headline)
        print(stars)

    for it in range(0, maxit):

        if it == 0:
            drfut = dr0

        else:

            def drfut(i, ss):
                if iid_process:
                    i = 0
                m = dp.node(i)
                l_ = lb(m, ss, p)
                u_ = ub(m, ss, p)
                x = eval_linear((sa0[i, :, 0],), xa0[i, :, 0], ss)[:, None]
                x = np.minimum(x, u_)
                x = np.maximum(x, l_)
                return x

        z[:, :, :] = 0

        for i_m in range(n_m):
            m = dp.node(i_m)
            for i_M in range(dp.n_inodes(i_m)):
                w = dp.iweight(i_m, i_M)
                M = dp.inode(i_m, i_M)
                S = gt(m, a, M, p)
                print(it, i_m, i_M)
                X = drfut(i_M, S)
                z[i_m, :, :] += w * h(M, S, X, p)
            xa[i_m, :, :] = τ(m, a, z[i_m, :, :], p)
            sa[i_m, :, :] = aτ(m, a, xa[i_m, :, :], p)

        if it > 1:
            η = abs(xa - xa0).max() + abs(sa - sa0).max()
        else:
            η = 1

        vprint("|{0:4} | {1:10.3e} |".format(it, η))

        if η < η_tol:
            break

        sa0[...] = sa
        xa0[...] = xa

    # resample the result on the standard grid
    endo_grid = grid["endo"]
    exo_grid = grid["exo"]
    mdr = DecisionRule(exo_grid, endo_grid, dprocess=dp, interp_method="cubic")

    mdr.set_values(
        np.concatenate([drfut(i, s)[None, :, :] for i in range(n_m)], axis=0)
    )

    sol = EGMResult(mdr, it, dp, (η < η_tol), η_tol, η)

    return sol

```

File: examples/models/consumption_savings_iid.yaml
```yaml
symbols:
    exogenous: [y]
    states: [w]
    expectations: [mr]
    poststates: [a]
    controls: [c]
    parameters: [β, γ, σ, ρ, r, cbar]


equations:

    transition:
        - w[t] = exp(y[t]) + (w[t-1]-c[t-1])*r

    arbitrage:
        - β*( c[t+1]/c[t] )^(-γ)*r - 1  | 0.0<=c[t]<=w[t]

    half_transition: |
        w[t] = exp(y[t]) + a[t-1]*r

    reverse_state: |
        w[t] = a[t] + c[t]

    expectation: |
        mr[t] = β*( c[t+1]/cbar )^(-γ)*r

    direct_response_egm: |
        c[t] = cbar*(mr[t])^(-1/γ)

calibration:
    β: 0.96
    γ: 4.0
    σ: 0.1
    ρ: 0.0
    r: 1.02
    cbar: c

    w: 1.0
    ξ: 0.0
    c: 0.9*w

domain:
    w: [0.01, 4.0]

exogenous: !UNormal
    sigma: σ


options:
    grid: !Cartesian
        orders: [100]

```

File: examples/models/rbc_ar1.yaml
```yaml
name: Real Business Cycle

symbols:

   exogenous: [z]
   states: [k]
   controls: [n, i]
   expectations: [m]
   values: [V]
   parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]
   rewards: [u]
   
definitions:
    y[t]: exp(z[t])*k[t]^alpha*n[t]^(1-alpha)
    c[t]: y[t] - i[t]
    rk[t]: alpha*y[t]/k[t]
    w[t]: (1-alpha)*y[t]/n[t]

equations:

    arbitrage:
        - chi*n[t]^eta*c[t]^sigma - w[t]                     ⟂ 0.0 <= n[t] <= inf
        - 1 - beta*(c[t]/c[t+1])^(sigma)*(1-delta+rk[t+1])   ⟂ 0.0 <= i[t] <= inf


    transition:
        - k[t] = (1-delta)*k[t-1] + i[t-1]

    value:
        - V[t] = c[t]^(1-sigma)/(1-sigma) - chi*n[t]^(1+eta)/(1+eta) + beta*V[t+1]

    felicity:
        - u[t] =  c[t]^(1-sigma)/(1-sigma) - chi*n[t]^(1+eta)/(1+eta)

    expectation:
        - m[t] = beta/c[t+1]^sigma*(1-delta+rk[t+1])

    direct_response:
        - n[t] = ((1-alpha)*exp(z[t])*k[t]^alpha*m[t]/chi)^(1/(eta+alpha))
        - i[t] = exp(z[t])*k[t]^alpha*n[t]^(1-alpha) - (m[t])^(-1/sigma)


calibration:

    # parameters
    beta: 0.99
    phi: 1
    delta : 0.025
    alpha : 0.33
    rho : 0.8
    sigma: 5
    eta: 1
    sig_z: 0.016
    zbar: 0
    chi : w/c^sigma/n^eta
    c_i: 1.5
    c_y: 0.5
    e_z: 0.0
    m: 0
    V0: (c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta))/(1-beta)

    # endogenous variables
    n: 0.33
    z: zbar
    rk: 1/beta-1+delta
    w: (1-alpha)*exp(z)*(k/n)^(alpha)
    k: n/(rk/alpha)^(1/(1-alpha))
    y: exp(z)*k^alpha*n^(1-alpha)
    i: delta*k
    c: y - i
    V: log(c)/(1-beta)
    u: c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)




exogenous: !VAR1
    rho: 0.8
    Sigma: [[sig_z^2]]

domain:
    k: [k*0.5, k*1.5]

options:
    grid: !Cartesian
        orders: [20]

```

File: examples/models_/open_economy_1d.yaml
```yaml
name: Open economy

# description: Adapted from "country portfolios dynamics"

symbols:
  states: [R]
  controls: [p_f, db_f]
  exogenous: [epsilon]
  parameters: [beta, gamma, ybar_1, ybar_2, kappa, dumb]


definitions:
    c_1: ybar_1 + R - db_f*p_f
    c_2: ybar_2 - R + db_f*p_f

equations:

    transition:
        - R = dumb*R(-1) + db_f(-1) + epsilon

    arbitrage:
        - beta*( c_1^(-kappa)*(c_1(1)/c_1)^(-gamma) + c_2^(-kappa)*(c_2(1)/c_2)^(-gamma) )/2 - p_f
        - beta*( c_1^(-kappa)*(c_1(1)/c_1)^(-gamma) - c_2^(-kappa)*(c_2(1)/c_2)^(-gamma) )        |  -inf <= db_f <= inf

############################
calibration:

    # controls
    p_f: beta
    db_f: 0

    # states
    R: 0

    #auxiliary
    c_1: ybar_1
    c_2: ybar_2

    epsilon: 0

    # parameters
    beta: 0.96
    gamma: 4.0

    ybar_1: 1.0
    ybar_2: 1.0
    kappa: 0.0
    dumb: 0.5


############################
exogenous: !Normal
  Sigma: [[ 0.05 ]]

domain:
    R: [-0.5, 0.5]
    
options:
    grid: !Cartesian
        orders: [10]

```

File: dolo/algos/results.py
```py
class AlgoResult:
    pass


from dataclasses import dataclass


@dataclass
class TimeIterationResult(AlgoResult):
    dr: object
    iterations: int
    complementarities: bool
    dprocess: object
    x_converged: bool
    x_tol: float
    err: float
    log: object  # TimeIterationLog
    trace: object  # {Nothing,IterationTrace}


@dataclass
class EGMResult(AlgoResult):
    dr: object
    iterations: int
    dprocess: object
    a_converged: bool
    a_tol: float
    err: float
    # log: object  # TimeIterationLog
    # trace: object  # {Nothing,IterationTrace}


@dataclass
class ValueIterationResult(AlgoResult):
    dr: object  #:AbstractDecisionRule
    drv: object  #:AbstractDecisionRule
    iterations: int
    dprocess: object  #:AbstractDiscretizedProcess
    x_converged: object  #:Bool
    x_tol: float
    x_err: float
    v_converged: bool
    v_tol: float
    v_err: float
    log: object  #:ValueIterationLog
    trace: object  #:Union{Nothing,IterationTrace}


@dataclass
class ImprovedTimeIterationResult(AlgoResult):
    dr: object  #:AbstractDecisionRule
    N: int
    f_x: float  #:Float64
    d_x: float  #:Float64
    x_converged: bool  #:Bool
    complementarities: bool  #:Bool
    # Time_search::
    radius: float  # :Float64
    trace_data: object
    L: object


@dataclass
class PerturbationResult(AlgoResult):
    dr: object  #:BiTaylorExpansion
    generalized_eigenvalues: object  # :Vector
    stable: bool  # biggest e.v. lam of solution is < 1
    determined: bool  # next eigenvalue is > lam + epsilon (MOD solution well defined)
    unique: bool  # next eigenvalue is > 1

```

File: dolo/algos/value_iteration.py
```py
import time
import numpy as np
import numpy
import scipy.optimize

from dolo.compiler.model import Model

from dolo.numeric.processes import DiscretizedIIDProcess

# from dolo.numeric.decision_rules_markov import MarkovDecisionRule, IIDDecisionRule
from dolo.numeric.decision_rule import DecisionRule, ConstantDecisionRule
from dolo.numeric.grids import Grid, CartesianGrid, SmolyakGrid, UnstructuredGrid
from dolo.misc.itprinter import IterationsPrinter


def constant_policy(model):
    return ConstantDecisionRule(model.calibration["controls"])


from .results import AlgoResult, ValueIterationResult


def value_iteration(
    model: Model,
    *,
    verbose: bool = False,  #
    details: bool = True,  #
    tol=1e-6,
    maxit=500,
    maxit_howard=20,
) -> ValueIterationResult:
    """
    Solve for the value function and associated Markov decision rule by iterating over
    the value function.

    Parameters:
    -----------
    model :
        model to be solved
    dr :
        decision rule to evaluate

    Returns:
    --------
    mdr : Markov decision rule
        The solved decision rule/policy function
    mdrv: decision rule
        The solved value function
    """

    transition = model.functions["transition"]
    felicity = model.functions["felicity"]
    controls_lb = model.functions["controls_lb"]
    controls_ub = model.functions["controls_ub"]

    parms = model.calibration["parameters"]
    discount = model.calibration["beta"]

    x0 = model.calibration["controls"]
    m0 = model.calibration["exogenous"]
    s0 = model.calibration["states"]
    r0 = felicity(m0, s0, x0, parms)

    process = model.exogenous

    grid, dprocess = model.discretize()
    endo_grid = grid["endo"]
    exo_grid = grid["exo"]

    n_ms = dprocess.n_nodes  # number of exogenous states
    n_mv = dprocess.n_inodes(0)  # this assume number of integration nodes is constant

    mdrv = DecisionRule(exo_grid, endo_grid)

    s = mdrv.endo_grid.nodes
    N = s.shape[0]
    n_x = len(x0)

    mdr = constant_policy(model)

    controls_0 = np.zeros((n_ms, N, n_x))
    for i_ms in range(n_ms):
        controls_0[i_ms, :, :] = mdr.eval_is(i_ms, s)

    values_0 = np.zeros((n_ms, N, 1))
    # for i_ms in range(n_ms):
    #     values_0[i_ms, :, :] = mdrv(i_ms, grid)

    mdr = DecisionRule(exo_grid, endo_grid)
    # mdr.set_values(controls_0)

    # THIRD: value function iterations until convergence
    it = 0
    err_v = 100
    err_v_0 = 0
    gain_v = 0.0
    err_x = 100
    err_x_0 = 0
    tol_x = 1e-5
    tol_v = 1e-7

    itprint = IterationsPrinter(
        ("N", int),
        ("Error_V", float),
        ("Gain_V", float),
        ("Error_x", float),
        ("Gain_x", float),
        ("Eval_n", int),
        ("Time", float),
        verbose=verbose,
    )
    itprint.print_header("Start value function iterations.")

    while (it < maxit) and (err_v > tol or err_x > tol_x):

        t_start = time.time()
        it += 1

        mdr.set_values(controls_0)
        if it > 2:
            ev = evaluate_policy(model, mdr, dr0=mdrv, verbose=False, details=True)
        else:
            ev = evaluate_policy(model, mdr, verbose=False, details=True)

        mdrv = ev.solution
        for i_ms in range(n_ms):
            values_0[i_ms, :, :] = mdrv.eval_is(i_ms, s)

        values = values_0.copy()
        controls = controls_0.copy()

        for i_m in range(n_ms):
            m = dprocess.node(i_m)
            for n in range(N):
                s_ = s[n, :]
                x = controls[i_m, n, :]
                lb = controls_lb(m, s_, parms)
                ub = controls_ub(m, s_, parms)
                bnds = [e for e in zip(lb, ub)]

                def valfun(xx):
                    return -choice_value(
                        transition,
                        felicity,
                        i_m,
                        s_,
                        xx,
                        mdrv,
                        dprocess,
                        parms,
                        discount,
                    )[0]

                res = scipy.optimize.minimize(valfun, x, bounds=bnds)
                controls[i_m, n, :] = res.x
                values[i_m, n, 0] = -valfun(x)

        # compute error, update value and dr
        err_x = abs(controls - controls_0).max()
        err_v = abs(values - values_0).max()
        t_end = time.time()
        elapsed = t_end - t_start

        values_0 = values
        controls_0 = controls

        gain_x = err_x / err_x_0
        gain_v = err_v / err_v_0

        err_x_0 = err_x
        err_v_0 = err_v

        itprint.print_iteration(
            N=it,
            Error_V=err_v,
            Gain_V=gain_v,
            Error_x=err_x,
            Gain_x=gain_x,
            Eval_n=ev.iterations,
            Time=elapsed,
        )

    itprint.print_finished()

    mdr = DecisionRule(exo_grid, endo_grid)

    mdr.set_values(controls)
    mdrv.set_values(values_0)

    if not details:
        return mdr, mdrv
    else:
        return ValueIterationResult(
            mdr,  #:AbstractDecisionRule
            mdrv,  #:AbstractDecisionRule
            it,  #:Int
            dprocess,  #:AbstractDiscretizedProcess
            err_x < tol_x,  #:Bool
            tol_x,  #:Float64
            err_x,  #:Float64
            err_v < tol_v,  #:Bool
            tol_v,  #:Float64
            err_v,  #:Float64
            None,  # log:     #:ValueIterationLog
            None,  # trace:   #:Union{Nothing,IterationTrace
        )


def choice_value(transition, felicity, i_ms, s, x, drv, dprocess, parms, beta):

    m = dprocess.node(i_ms)
    cont_v = 0.0
    for I_ms in range(dprocess.n_inodes(i_ms)):
        M = dprocess.inode(i_ms, I_ms)
        prob = dprocess.iweight(i_ms, I_ms)
        S = transition(m, s, x, M, parms)
        V = drv(I_ms, S)[0]
        cont_v += prob * V
    return felicity(m, s, x, parms) + beta * cont_v


class EvaluationResult:
    def __init__(self, solution, iterations, tol, error):
        self.solution = solution
        self.iterations = iterations
        self.tol = tol
        self.error = error


def evaluate_policy(
    model,
    mdr,
    tol=1e-8,
    maxit=2000,
    grid={},
    verbose=True,
    dr0=None,
    hook=None,
    integration_orders=None,
    details=False,
    interp_method="cubic",
):
    """Compute value function corresponding to policy ``dr``

    Parameters:
    -----------

    model:
        "dtcscc" model. Must contain a 'value' function.

    mdr:
        decision rule to evaluate

    Returns:
    --------

    decision rule:
        value function (a function of the space similar to a decision rule
        object)

    """

    process = model.exogenous
    grid, dprocess = model.discretize()
    endo_grid = grid["endo"]
    exo_grid = grid["exo"]

    n_ms = dprocess.n_nodes  # number of exogenous states
    n_mv = dprocess.n_inodes(0)  # this assume number of integration nodes is constant

    x0 = model.calibration["controls"]
    v0 = model.calibration["values"]
    parms = model.calibration["parameters"]
    n_x = len(x0)
    n_v = len(v0)
    n_s = len(model.symbols["states"])

    if dr0 is not None:
        mdrv = dr0
    else:
        mdrv = DecisionRule(exo_grid, endo_grid, interp_method=interp_method)

    s = mdrv.endo_grid.nodes
    N = s.shape[0]

    if isinstance(mdr, np.ndarray):
        controls = mdr
    else:
        controls = np.zeros((n_ms, N, n_x))
        for i_m in range(n_ms):
            controls[i_m, :, :] = mdr.eval_is(i_m, s)

    values_0 = np.zeros((n_ms, N, n_v))
    if dr0 is None:
        for i_m in range(n_ms):
            values_0[i_m, :, :] = v0[None, :]
    else:
        for i_m in range(n_ms):
            values_0[i_m, :, :] = dr0.eval_is(i_m, s)

    val = model.functions["value"]
    g = model.functions["transition"]

    sh_v = values_0.shape

    err = 10
    inner_maxit = 50
    it = 0

    if verbose:
        headline = "|{0:^4} | {1:10} | {2:8} | {3:8} |".format(
            "N", " Error", "Gain", "Time"
        )
        stars = "-" * len(headline)
        print(stars)
        print(headline)
        print(stars)

    t1 = time.time()

    err_0 = np.nan

    verbit = verbose == "full"

    while err > tol and it < maxit:

        it += 1

        t_start = time.time()

        mdrv.set_values(values_0.reshape(sh_v))
        values = update_value(
            val, g, s, controls, values_0, mdr, mdrv, dprocess, parms
        ).reshape((-1, n_v))
        err = abs(values.reshape(sh_v) - values_0).max()

        err_SA = err / err_0
        err_0 = err

        values_0 = values.reshape(sh_v)

        t_finish = time.time()
        elapsed = t_finish - t_start

        if verbose:
            print(
                "|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} |".format(
                    it, err, err_SA, elapsed
                )
            )

    # values_0 = values.reshape(sh_v)

    t2 = time.time()

    if verbose:
        print(stars)
        print("Elapsed: {} seconds.".format(t2 - t1))
        print(stars)

    if not details:
        return mdrv
    else:
        return EvaluationResult(mdrv, it, tol, err)


def update_value(val, g, s, x, v, dr, drv, dprocess, parms):

    N = s.shape[0]
    n_s = s.shape[1]

    n_ms = dprocess.n_nodes  # number of exogenous states
    n_mv = dprocess.n_inodes(0)  # this assume number of integration nodes is constant

    res = np.zeros_like(v)

    for i_ms in range(n_ms):

        m = dprocess.node(i_ms)[None, :].repeat(N, axis=0)

        xm = x[i_ms, :, :]
        vm = v[i_ms, :, :]

        for I_ms in range(n_mv):

            # M = P[I_ms,:][None,:]
            M = dprocess.inode(i_ms, I_ms)[None, :].repeat(N, axis=0)
            prob = dprocess.iweight(i_ms, I_ms)

            S = g(m, s, xm, M, parms)
            XM = dr.eval_ijs(i_ms, I_ms, S)
            VM = drv.eval_ijs(i_ms, I_ms, S)
            rr = val(m, s, xm, vm, M, S, XM, VM, parms)

            res[i_ms, :, :] += prob * rr

    return res

```

File: dolo/algos/perfect_foresight.py
```py
import numpy as np
import pandas as pd
from numpy import array, atleast_2d, linspace, zeros
from scipy.optimize import root

from dolo.compiler.model import Model
from dolo.numeric.optimize.ncpsolve import ncpsolve


def _shocks_to_epsilons(model, shocks, T):
    """
    Helper function to support input argument `shocks` being one of many
    different data types. Will always return a `T, n_e` matrix.
    """
    n_e = len(model.calibration["exogenous"])

    # if we have a DataFrame, convert it to a dict and rely on the method below
    if isinstance(shocks, pd.DataFrame):
        shocks = {k: shocks[k].tolist() for k in shocks.columns}

    # handle case where shocks might be a dict. Be careful to handle case where
    # value arrays are not the same length
    if isinstance(shocks, dict):
        epsilons = np.zeros((T + 1, n_e))
        for i, k in enumerate(model.symbols["exogenous"]):
            if k in shocks:
                this_shock = shocks[k]
                epsilons[: len(this_shock), i] = this_shock
                epsilons[len(this_shock) :, i] = this_shock[-1]
            else:
                # otherwise set to value in calibration
                epsilons[:, i] = model.calibration["exogenous"][i]

        return epsilons

    # read from calibration if not given
    if shocks is None:
        shocks = model.calibration["exogenous"]

    # now we just assume that shocks is array-like and try using the output of
    # np.asarray(shocks)
    shocks = np.asarray(shocks)
    shocks = shocks.reshape((-1, n_e))

    # until last period, exogenous shock takes its last value
    epsilons = np.zeros((T + 1, n_e))
    epsilons[: (shocks.shape[0] - 1), :] = shocks[1:, :]
    epsilons[(shocks.shape[0] - 1) :, :] = shocks[-1:, :]

    return epsilons


def deterministic_solve(
    model: Model,
    *,  #
    verbose: bool = True,  #
    ignore_constraints: bool = False,  #
    exogenous=None,
    s0=None,
    m0=None,
    T=100,
    maxit=100,
    initial_guess=None,
    solver="ncpsolve",
    keep_steady_state=False,
    s1=None,  # deprecated
    shocks=None,  # deprecated
    tol=1e-6,
):
    """
    Computes a perfect foresight simulation using a stacked-time algorithm.

    Typical simulation exercises are:
    - start from an out-of-equilibrium exogenous and/or endogenous state: specify `s0` and or `m0`. Missing values are taken from the calibration (`model.calibration`).
    - specify an exogenous path for shocks `exogenous`. Initial exogenous state `m0` is then first value of exogenous values. Economy is supposed to have been at the equilibrium for $t<0$, which pins
    down initial endogenous state `s0`. `x0` is a jump variable.

    If $s0$ is not specified it is then set
    equal to the steady-state consistent with the first value

    The initial state is specified either by providing a series of exogenous
    shocks and assuming the model is initially in equilibrium with the first
    value of the shock, or by specifying an initial value for the states.

    Parameters
    ----------
    model : Model
        Model to be solved
    exogenous : array-like, dict, or pandas.DataFrame
        A specification for the path of exogenous variables (aka shocks). Can be any of the
        following (note by "declaration order" below we mean the order
        of `model.symbols["exogenous"]`):

        - A 1d numpy array-like specifying a time series for a single
          exogenous variable, or all exogenous variables stacked into a single array.
        - A 2d numpy array where each column specifies the time series
          for one of the shocks in declaration order. This must be an
          `N` by number of shocks 2d array.
        - A dict where keys are strings found in
          `model.symbols["exogenous"]` and values are a time series of
          values for that shock. For exogenous variables that do not appear in
          this dict, the shock is set to the calibrated value. Note
          that this interface is the most flexible as it allows the user
          to pass values for only a subset of the model shocks and it
          allows the passed time series to be of different lengths.
        - A DataFrame where columns map shock names into time series.
          The same assumptions and behavior that are used in the dict
          case apply here

        If nothing is given here, `exogenous` is set equal to the
        calibrated values found in `model.calibration["exogenous"]` for
        all periods.

        If the length of any time-series in shocks is less than `T`
        (see below) it is assumed that that particular shock will
        remain at the final given value for the duration of the
        simulation.
    s0 : None or ndarray or dict
        If vector with the value of initial states
        If an exogenous timeseries is given for exogenous shocks, `s0` will be computed as the steady-state value that is consistent with its first value.

    T : int
        horizon for the perfect foresight simulation
    maxit : int
        maximum number of iteration for the nonlinear solver
    verbose : boolean
        if True, the solver displays iterations
    tol : float
        stopping criterium for the nonlinear solver
    ignore_constraints : bool
        if True, complementarity constraints are ignored.
    keep_steady_state : bool
        if True, initial steady-states and steady-controls are appended to the simulation with date -1.
    Returns
    -------
    pandas dataframe
        a dataframe with T+1 observations of the model variables along the
        simulation (states, controls, auxiliaries). The  simulation should return to a steady-state
        consistent with the last specified value of the exogenous shocks.

    """

    if shocks is not None:
        import warnings

        warnings.warn("`shocks` argument is deprecated. Use `exogenous` instead.")
        exogenous = shocks

    if s1 is not None:
        import warnings

        warnings.warn("`s1` argument is deprecated. Use `s0` instead.")
        s0 = s1

    # definitions
    n_s = len(model.calibration["states"])
    n_x = len(model.calibration["controls"])
    p = model.calibration["parameters"]

    if exogenous is not None:
        epsilons = _shocks_to_epsilons(model, exogenous, T)
        m0 = epsilons[0, :]
        # get initial steady-state
        from dolo.algos.steady_state import find_steady_state

        start_state = find_steady_state(model, m=m0)
        s0 = start_state["states"]
        x0 = start_state["controls"]
        m1 = epsilons[1, :]
        s1 = model.functions["transition"](m0, s0, x0, m1, p)
    else:
        if s0 is None:
            s0 = model.calibration["states"]
        if m0 is None:
            m0 = model.calibration["exogenous"]
        # if m0 is None:
        #     m0 = np.zeros(len(model.symbols['exogenous']))
        # we should probably do something here with the nature of the exogenous process if specified
        # i.e. compute nonlinear irf
        epsilons = _shocks_to_epsilons(model, exogenous, T)
        x0 = model.calibration["controls"]
        m1 = epsilons[1, :]
        s1 = model.functions["transition"](m0, s0, x0, m1, p)
        s1 = np.array(s1)

    x1_g = model.calibration["controls"]  # we can do better here
    sT_g = model.calibration["states"]  # we can do better here
    xT_g = model.calibration["controls"]  # we can do better here

    if initial_guess is None:
        start = np.concatenate([s1, x1_g])
        final = np.concatenate([sT_g, xT_g])
        initial_guess = np.row_stack(
            [start * (1 - l) + final * l for l in linspace(0.0, 1.0, T + 1)]
        )

    else:
        if isinstance(initial_guess, pd.DataFrame):
            initial_guess = np.array(
                initial_guess[model.symbols["states"] + model.symbols["controls"]]
            )
        initial_guess = initial_guess[1:, :]
        initial_guess = initial_guess[:, : n_s + n_x]

    sh = initial_guess.shape

    if model.x_bounds and not ignore_constraints:
        initial_states = initial_guess[:, :n_s]
        [lb, ub] = [u(epsilons[:, :], initial_states, p) for u in model.x_bounds]
        lower_bound = initial_guess * 0 - np.inf
        lower_bound[:, n_s:] = lb
        upper_bound = initial_guess * 0 + np.inf
        upper_bound[:, n_s:] = ub
        test1 = max(lb.max(axis=0) - lb.min(axis=0))
        test2 = max(ub.max(axis=0) - ub.min(axis=0))
        if test1 > 0.00000001 or test2 > 0.00000001:
            msg = "Not implemented: perfect foresight solution requires that "
            msg += "controls have constant bounds."
            raise Exception(msg)
    else:
        ignore_constraints = True
        lower_bound = None
        upper_bound = None

    det_residual(model, initial_guess, s1, xT_g, epsilons)

    if not ignore_constraints:

        def ff(vec):
            return det_residual(
                model, vec.reshape(sh), s1, xT_g, epsilons, jactype="sparse"
            )

        v0 = initial_guess.ravel()
        if solver == "ncpsolve":
            sol, nit = ncpsolve(
                ff,
                lower_bound.ravel(),
                upper_bound.ravel(),
                initial_guess.ravel(),
                verbose=verbose,
                maxit=maxit,
                tol=tol,
                jactype="sparse",
            )
        else:
            from dolo.numeric.extern.lmmcp import lmmcp

            sol = lmmcp(
                lambda u: ff(u)[0],
                lambda u: ff(u)[1].todense(),
                lower_bound.ravel(),
                upper_bound.ravel(),
                initial_guess.ravel(),
                verbose=verbose,
            )
            nit = -1

        sol = sol.reshape(sh)

    else:

        def ff(vec):
            ll = det_residual(model, vec.reshape(sh), s1, xT_g, epsilons, diff=True)
            return ll

        v0 = initial_guess.ravel()
        # from scipy.optimize import root
        # sol = root(ff, v0, jac=True)
        # sol = sol.x.reshape(sh)
        from dolo.numeric.optimize.newton import newton

        sol, nit = newton(ff, v0, jactype="sparse")
        sol = sol.reshape(sh)

    # sol = sol[:-1, :]

    if (exogenous is not None) and keep_steady_state:
        sx = np.concatenate([s0, x0])
        sol = np.concatenate([sx[None, :], sol], axis=0)
        epsilons = np.concatenate([epsilons[:1:], epsilons], axis=0)
        index = range(-1, T + 1)
    else:
        index = range(0, T + 1)
    # epsilons = np.concatenate([epsilons[:1,:], epsilons], axis=0)

    if "auxiliary" in model.functions:
        colnames = (
            model.symbols["states"]
            + model.symbols["controls"]
            + model.symbols["auxiliaries"]
        )
        # compute auxiliaries
        y = model.functions["auxiliary"](epsilons, sol[:, :n_s], sol[:, n_s:], p)
        sol = np.column_stack([sol, y])
    else:
        colnames = model.symbols["states"] + model.symbols["controls"]

    sol = np.column_stack([sol, epsilons])
    colnames = colnames + model.symbols["exogenous"]

    ts = pd.DataFrame(sol, columns=colnames, index=index)
    return ts


def det_residual(model, guess, start, final, shocks, diff=True, jactype="sparse"):
    """
    Computes the residuals, the derivatives of the stacked-time system.
    :param model: an fga model
    :param guess: the guess for the simulated values. An `(n_s.n_x) x N` array,
                  where n_s is the number of states,
    n_x the number of controls, and `N` the length of the simulation.
    :param start: initial boundary condition (initial value of the states)
    :param final: final boundary condition (last value of the controls)
    :param shocks: values for the exogenous shocks
    :param diff: if True, the derivatives are computes
    :return: a list with two elements:
        - an `(n_s.n_x) x N` array with the residuals of the system
        - a `(n_s.n_x) x N x (n_s.n_x) x N` array representing the jacobian of
             the system
    """

    # TODO: compute a sparse derivative and ensure the solvers can deal with it

    n_s = len(model.symbols["states"])
    n_x = len(model.symbols["controls"])

    # n_e = len(model.symbols['shocks'])
    N = guess.shape[0]

    p = model.calibration["parameters"]

    f = model.functions["arbitrage"]
    g = model.functions["transition"]

    vec = guess[:-1, :]
    vec_f = guess[1:, :]

    s = vec[:, :n_s]
    x = vec[:, n_s:]
    S = vec_f[:, :n_s]
    X = vec_f[:, n_s:]

    m = shocks[:-1, :]
    M = shocks[1:, :]

    if diff:
        SS, SS_m, SS_s, SS_x, SS_M = g(m, s, x, M, p, diff=True)
        R, R_m, R_s, R_x, R_M, R_S, R_X = f(m, s, x, M, S, X, p, diff=True)
    else:
        SS = g(m, s, x, M, p)
        R = f(m, s, x, M, S, X, p)

    res_s = SS - S
    res_x = R

    res = np.zeros((N, n_s + n_x))

    res[1:, :n_s] = res_s
    res[:-1, n_s:] = res_x

    res[0, :n_s] = -(guess[0, :n_s] - start)
    res[-1, n_s:] = -(guess[-1, n_s:] - guess[-2, n_s:])

    if not diff:
        return res
    else:

        sparse_jac = False
        if not sparse_jac:

            # we compute the derivative matrix
            res_s_s = SS_s
            res_s_x = SS_x

            # next block is probably very inefficient
            jac = np.zeros((N, n_s + n_x, N, n_s + n_x))
            for i in range(N - 1):
                jac[i, n_s:, i, :n_s] = R_s[i, :, :]
                jac[i, n_s:, i, n_s:] = R_x[i, :, :]
                jac[i, n_s:, i + 1, :n_s] = R_S[i, :, :]
                jac[i, n_s:, i + 1, n_s:] = R_X[i, :, :]
                jac[i + 1, :n_s, i, :n_s] = SS_s[i, :, :]
                jac[i + 1, :n_s, i, n_s:] = SS_x[i, :, :]
                jac[i + 1, :n_s, i + 1, :n_s] = -np.eye(n_s)
                # jac[i,n_s:,i,:n_s] = R_s[i,:,:]
                # jac[i,n_s:,i,n_s:] = R_x[i,:,:]
                # jac[i+1,n_s:,i,:n_s] = R_S[i,:,:]
                # jac[i+1,n_s:,i,n_s:] = R_X[i,:,:]
                # jac[i,:n_s,i+1,:n_s] = SS_s[i,:,:]
                # jac[i,:n_s,i+1,n_s:] = SS_x[i,:,:]
                # jac[i+1,:n_s,i+1,:n_s] = -np.eye(n_s)
            jac[0, :n_s, 0, :n_s] = -np.eye(n_s)
            jac[-1, n_s:, -1, n_s:] = -np.eye(n_x)
            jac[-1, n_s:, -2, n_s:] = +np.eye(n_x)
            nn = jac.shape[0] * jac.shape[1]
            res = res.ravel()
            jac = jac.reshape((nn, nn))

        if jactype == "sparse":
            from scipy.sparse import csc_matrix, csr_matrix

            jac = csc_matrix(jac)
            # scipy bug ? I don't get the same with csr

        return [res, jac]

```

File: dolo/algos/commands.py
```py
# this is for compatibility purposes only

from dolo.algos.time_iteration import time_iteration
from dolo.algos.perfect_foresight import deterministic_solve
from dolo.algos.simulations import simulate, response, tabulate, plot_decision_rule
from dolo.algos.value_iteration import evaluate_policy, value_iteration
from dolo.algos.improved_time_iteration import improved_time_iteration
from dolo.algos.steady_state import residuals
from dolo.algos.perturbation import perturb
from dolo.algos.ergodic import ergodic_distribution

approximate_controls = perturb

```

File: dolo/numeric/decision_rule.py
```py
import numpy
from numpy import array, zeros
from interpolation.smolyak import SmolyakGrid as SmolyakGrid0
from interpolation.smolyak import SmolyakInterp, build_B
from dolo.numeric.grids import cat_grids, n_nodes, node
from dolo.numeric.grids import UnstructuredGrid, CartesianGrid, SmolyakGrid, EmptyGrid
from dolo.numeric.misc import mlinspace
import scipy
from dolo.numeric.grids import *

# from dolo.numeric.decision_rule import CallableDecisionRule, cat_grids
import numpy as np

import numpy as np


def filter_controls(a, b, ndims, controls):

    from interpolation.splines.filter_cubic import filter_data, filter_mcoeffs

    dinv = (b - a) / (ndims - 1)
    ndims = array(ndims)
    n_m, N, n_x = controls.shape
    coefs = zeros((n_m,) + tuple(ndims + 2) + (n_x,))
    for i_m in range(n_m):
        tt = filter_mcoeffs(a, b, ndims, controls[i_m, ...])
        # for i_x in range(n_x):
        coefs[i_m, ...] = tt
    return coefs


class Linear:
    pass


class Cubic:
    pass


class Chebychev:
    pass


interp_methods = {
    "cubic": Cubic(),
    "linear": Linear(),
    "multilinear": Linear(),
    "chebychev": Chebychev(),
}

###


class CallableDecisionRule:
    def __call__(self, *args):
        args = [np.array(e) for e in args]
        if len(args) == 1:
            if args[0].ndim == 1:
                return self.eval_s(args[0][None, :])[0, :]
            else:
                return self.eval_s(args[0])
        elif len(args) == 2:
            if args[0].dtype in ("int64", "int32"):
                (i, s) = args
                if s.ndim == 1:
                    return self.eval_is(i, s[None, :])[0, :]
                else:
                    return self.eval_is(i, s)
                return self.eval_is()
            else:
                (m, s) = args[0], args[1]
                if s.ndim == 1 and m.ndim == 1:
                    return self.eval_ms(m[None, :], s[None, :])[0, :]
                elif m.ndim == 1:
                    m = m[None, :]
                elif s.ndim == 1:
                    s = s[None, :]
                return self.eval_ms(m, s)


class DecisionRule(CallableDecisionRule):

    exo_grid: Grid
    endo_grid: Grid

    def __init__(
        self,
        exo_grid: Grid,
        endo_grid: Grid,
        interp_method="cubic",
        dprocess=None,
        values=None,
    ):

        if interp_method not in interp_methods.keys():
            raise Exception(
                f"Unknown interpolation type: {interp_method}. Try one of: {tuple(interp_methods.keys())}"
            )

        self.exo_grid = exo_grid
        self.endo_grid = endo_grid
        self.interp_method = interp_method
        self.dprocess = dprocess

        self.__interp_method__ = interp_methods[interp_method]

        # here we could replace with a caching mechanism resolving dispatch in advance
        self.__eval_ms__ = eval_ms
        self.__eval_is__ = eval_is
        self.__eval_s__ = eval_s
        self.__get_coefficients__ = get_coefficients

        if values is not None:
            self.set_values(values)

    def set_values(self, x):
        self.coefficients = self.__get_coefficients__(
            self, self.exo_grid, self.endo_grid, self.__interp_method__, x
        )

    def eval_ms(self, m, s):
        return self.__eval_ms__(
            self, self.exo_grid, self.endo_grid, self.__interp_method__, m, s
        )

    def eval_is(self, i, s):
        return self.__eval_is__(
            self, self.exo_grid, self.endo_grid, self.__interp_method__, i, s
        )

    def eval_s(self, s):
        return self.__eval_s__(
            self, self.exo_grid, self.endo_grid, self.__interp_method__, s
        )

    def eval_ijs(self, i, j, s):

        if isinstance(self.exo_grid, UnstructuredGrid):
            out = self.eval_is(j, s)
        elif isinstance(self.exo_grid, EmptyGrid):
            out = self.eval_s(s)
        elif isinstance(self.exo_grid, CartesianGrid):
            m = self.dprocess.inode(i, j)[None, :].repeat(s.shape[0], axis=0)
            out = self.eval_ms(m, s)
        else:
            raise Exception("Not Implemented.")

        return out


# this is *not* meant to be used by users

from multipledispatch import dispatch

namespace = dict()
multimethod = dispatch(namespace=namespace)

# Cartesian x Cartesian x Linear


@multimethod
def get_coefficients(
    itp: object,
    exo_grid: CartesianGrid,
    endo_grid: CartesianGrid,
    interp_type: Linear,
    x: object,
):
    grid = exo_grid + endo_grid
    xx = x.reshape(tuple(grid.n) + (-1,))
    return xx.copy()


@multimethod
def eval_ms(
    itp: object,
    exo_grid: CartesianGrid,
    endo_grid: CartesianGrid,
    interp_type: Linear,
    m: object,
    s: object,
):

    assert m.ndim == s.ndim == 2

    grid = exo_grid + endo_grid  # one single CartesianGrid

    coeffs = itp.coefficients

    gg = grid.__numba_repr__()
    from interpolation.splines import eval_linear

    x = np.concatenate([m, s], axis=1)

    return eval_linear(gg, coeffs, x)


@multimethod
def eval_is(
    itp: object,
    exo_grid: CartesianGrid,
    endo_grid: CartesianGrid,
    interp_type: Linear,
    i: object,
    s: object,
):
    m = exo_grid.node(i)[None, :]
    return eval_ms(itp, exo_grid, endo_grid, interp_type, m, s)


# Cartesian x Cartesian x Cubic


@multimethod
def get_coefficients(
    itp: object,
    exo_grid: CartesianGrid,
    endo_grid: CartesianGrid,
    interp_type: Cubic,
    x: object,
):

    from interpolation.splines.prefilter_cubic import prefilter_cubic

    grid = exo_grid + endo_grid  # one single CartesianGrid
    x = x.reshape(tuple(grid.n) + (-1,))
    gg = grid.__numba_repr__()
    return prefilter_cubic(gg, x)


@multimethod
def eval_ms(
    itp: object,
    exo_grid: CartesianGrid,
    endo_grid: CartesianGrid,
    interp_type: Cubic,
    m: object,
    s: object,
):

    from interpolation.splines import eval_cubic

    assert m.ndim == s.ndim == 2

    grid = exo_grid + endo_grid  # one single CartesianGrid
    coeffs = itp.coefficients

    gg = grid.__numba_repr__()

    x = np.concatenate([m, s], axis=1)

    return eval_cubic(gg, coeffs, x)


@multimethod
def eval_is(
    itp: object,
    exo_grid: CartesianGrid,
    endo_grid: CartesianGrid,
    interp_type: Cubic,
    i: object,
    s: object,
):
    m = exo_grid.node(i)[None, :]
    return eval_ms(itp, exo_grid, endo_grid, interp_type, m, s)


# UnstructuredGrid x Cartesian x Linear


@multimethod
def get_coefficients(
    itp: object,
    exo_grid: UnstructuredGrid,
    endo_grid: CartesianGrid,
    interp_type: Linear,
    x: object,
):
    return [x[i].reshape(tuple(endo_grid.n) + (-1,)).copy() for i in range(x.shape[0])]


@multimethod
def eval_is(
    itp: object,
    exo_grid: UnstructuredGrid,
    endo_grid: CartesianGrid,
    interp_type: Linear,
    i: object,
    s: object,
):

    from interpolation.splines import eval_linear

    assert s.ndim == 2
    coeffs = itp.coefficients[i]
    gg = endo_grid.__numba_repr__()

    return eval_linear(gg, coeffs, s)


# UnstructuredGrid x Cartesian x Cubic


@multimethod
def get_coefficients(
    itp: object,
    exo_grid: UnstructuredGrid,
    endo_grid: CartesianGrid,
    interp_type: Cubic,
    x: object,
):
    from interpolation.splines.prefilter_cubic import prefilter_cubic

    gg = endo_grid.__numba_repr__()
    return [
        prefilter_cubic(gg, x[i].reshape(tuple(endo_grid.n) + (-1,)))
        for i in range(x.shape[0])
    ]


@multimethod
def eval_is(
    itp: object,
    exo_grid: UnstructuredGrid,
    endo_grid: CartesianGrid,
    interp_type: Cubic,
    i: object,
    s: object,
):

    from interpolation.splines import eval_cubic

    assert s.ndim == 2
    coeffs = itp.coefficients[i]
    gg = endo_grid.__numba_repr__()
    return eval_cubic(gg, coeffs, s)


# UnstructuredGrid x Cartesian x Linear


@multimethod
def get_coefficients(
    itp: object,
    exo_grid: UnstructuredGrid,
    endo_grid: CartesianGrid,
    interp_type: Linear,
    x: object,
):
    return [x[i].copy() for i in range(x.shape[0])]


@multimethod
def eval_is(
    itp: object,
    exo_grid: UnstructuredGrid,
    endo_grid: CartesianGrid,
    interp_type: Linear,
    i: object,
    s: object,
):

    from interpolation.splines import eval_linear

    assert s.ndim == 2

    coeffs = itp.coefficients[i]
    gg = endo_grid.__numba_repr__()

    return eval_linear(gg, coeffs, s)


# Empty x Cartesian x Linear


@multimethod
def get_coefficients(
    itp: object,
    exo_grid: EmptyGrid,
    endo_grid: CartesianGrid,
    interp_type: Linear,
    x: object,
):
    grid = exo_grid + endo_grid
    xx = x.reshape(tuple(grid.n) + (-1,))
    return xx.copy()


@multimethod
def eval_s(
    itp: object,
    exo_grid: EmptyGrid,
    endo_grid: CartesianGrid,
    interp_type: Linear,
    s: object,
):
    from interpolation.splines import eval_linear

    assert s.ndim == 2
    coeffs = itp.coefficients
    gg = endo_grid.__numba_repr__()
    return eval_linear(gg, coeffs, s)


# Empty x Cartesian x Cubic


@multimethod
def get_coefficients(
    itp: object,
    exo_grid: EmptyGrid,
    endo_grid: CartesianGrid,
    interp_type: Cubic,
    x: object,
):
    from interpolation.splines.prefilter_cubic import prefilter_cubic

    grid = endo_grid  # one single CartesianGrid
    gg = endo_grid.__numba_repr__()
    return prefilter_cubic(gg, x[0].reshape(tuple(grid.n) + (-1,)))


@multimethod
def eval_s(
    itp: object,
    exo_grid: EmptyGrid,
    endo_grid: CartesianGrid,
    interp_type: Cubic,
    s: object,
):
    from interpolation.splines import eval_cubic

    assert s.ndim == 2
    coeffs = itp.coefficients
    gg = endo_grid.__numba_repr__()
    return eval_cubic(gg, coeffs, s)


## an empty grid can be indexed by an integer or a vector


@multimethod
def eval_is(
    itp: object,
    exo_grid: EmptyGrid,
    endo_grid: object,
    interp_type: object,
    i: object,
    s: object,
):
    return eval_s(itp, exo_grid, endo_grid, interp_type, s)


@multimethod
def eval_ms(
    itp: object,
    exo_grid: EmptyGrid,
    endo_grid: object,
    interp_type: object,
    m: object,
    s: object,
):
    return eval_s(itp, exo_grid, endo_grid, interp_type, s)


####


class ConstantDecisionRule(CallableDecisionRule):
    def __init__(self, x0):
        self.x0 = x0

    def eval_s(self, s):
        if s.ndim == 1:
            return self.x0
        else:
            N = s.shape[0]
            return self.x0[None, :].repeat(N, axis=0)

    def eval_is(self, i, s):
        return self.eval_s(s)

    def eval_ms(self, m, s):
        return self.eval_s(s)


import dolang
import dolang.symbolic
from dolang.symbolic import stringify_symbol
from dolo.numeric.decision_rule import CallableDecisionRule
from dolang.factory import FlatFunctionFactory
from dolang.function_compiler import make_method_from_factory


class CustomDR(CallableDecisionRule):
    def __init__(self, values, model=None):

        from dolang.symbolic import sanitize, stringify

        exogenous = model.symbols["exogenous"]
        states = model.symbols["states"]
        controls = model.symbols["controls"]
        parameters = model.symbols["parameters"]

        preamble = dict([(s, values[s]) for s in values.keys() if s not in controls])
        equations = [values[s] for s in controls]

        variables = exogenous + states + controls + [*preamble.keys()]

        preamble_str = dict()

        for k in [*preamble.keys()]:
            v = preamble[k]
            if "(" not in k:
                vv = f"{k}(0)"
            else:
                vv = k

            preamble_str[stringify(vv)] = stringify(sanitize(v, variables))

        # let's reorder the preamble
        from dolang.triangular_solver import get_incidence, triangular_solver

        incidence = get_incidence(preamble_str)
        sol = triangular_solver(incidence)
        kk = [*preamble_str.keys()]
        preamble_str = dict([(kk[k], preamble_str[kk[k]]) for k in sol])

        equations = [
            dolang.symbolic.sanitize(eq, variables=variables) for eq in equations
        ]
        equations_strings = [dolang.stringify(eq) for eq in equations]

        args = dict(
            [
                ("m", [(e, 0) for e in exogenous]),
                ("s", [(e, 0) for e in states]),
                ("p", [e for e in parameters]),
            ]
        )

        args = dict([(k, [stringify_symbol(e) for e in v]) for k, v in args.items()])

        targets = [stringify_symbol((e, 0)) for e in controls]

        eqs = dict([(targets[i], eq) for i, eq in enumerate(equations_strings)])

        fff = FlatFunctionFactory(preamble_str, eqs, args, "custom_dr")

        fun, gufun = make_method_from_factory(fff)

        self.p = model.calibration["parameters"]
        self.exo_grid = model.exogenous.discretize()  # this is never used
        self.endo_grid = model.endo_grid
        self.gufun = gufun

    def eval_ms(self, m, s):

        return self.gufun(m, s, self.p)

```

File: dolo/compiler/model.py
```py
from dolang.symbolic import sanitize, parse_string, str_expression
from dolang.language import eval_data
from dolang.symbolic import str_expression

import copy


class SymbolicModel:
    def __init__(self, data):

        self.data = data

    @property
    def symbols(self):

        if self.__symbols__ is None:

            from .misc import LoosyDict, equivalent_symbols
            from dolang.symbolic import remove_timing, parse_string, str_expression

            symbols = LoosyDict(equivalences=equivalent_symbols)
            for sg in self.data["symbols"].keys():
                symbols[sg] = [s.value for s in self.data["symbols"][sg]]

            self.__symbols__ = symbols

            # the following call adds auxiliaries (tricky, isn't it?)
            self.definitions

        return self.__symbols__

    @property
    def variables(self):
        if self.__variables__ is None:

            self.__variables__ = sum(
                [self.symbols[e] for e in self.symbols.keys() if e != "parameters"], []
            )

        return self.__variables__

    @property
    def equations(self):
        import yaml.nodes

        if self.__equations__ is None:

            vars = self.variables + [*self.definitions.keys()]

            d = dict()
            for g, v in self.data["equations"].items():

                # new style
                if isinstance(v, yaml.nodes.ScalarNode):
                    assert v.style == "|"
                    if g in ("arbitrage",):
                        start = "complementarity_block"
                    else:
                        start = "assignment_block"
                    eqs = parse_string(v, start=start)
                    eqs = sanitize(eqs, variables=vars)
                    eq_list = eqs.children
                # old style
                else:
                    eq_list = []
                    for eq_string in v:
                        start = "equation"  # it should be assignment
                        eq = parse_string(eq_string, start=start)
                        eq = sanitize(eq, variables=vars)
                        eq_list.append(eq)

                if g in ("arbitrage",):
                    ll = []  # List[str]
                    ll_lb = []  # List[str]
                    ll_ub = []  # List[str]
                    with_complementarity = False
                    for i, eq in enumerate(eq_list):
                        if eq.data == "double_complementarity":
                            v = eq.children[1].children[1].children[0].children[0].value
                            t = int(
                                eq.children[1].children[1].children[1].children[0].value
                            )
                            expected = (
                                self.symbols["controls"][i],
                                0,
                            )  # TODO raise nice error message
                            if (v, t) != expected:
                                raise Exception(
                                    f"Incorrect variable in complementarity: expected {expected}. Found {(v,t)}"
                                )
                            ll_lb.append(str_expression(eq.children[1].children[0]))
                            ll_ub.append(str_expression(eq.children[1].children[2]))
                            eq = eq.children[0]
                            with_complementarity = True
                        else:
                            ll_lb.append("-inf")
                            ll_ub.append("inf")
                        from dolang.symbolic import list_symbols

                        # syms = list_symbols(eq)
                        ll.append(str_expression(eq))
                    d[g] = ll
                    if with_complementarity:
                        d[g + "_lb"] = ll_lb
                        d[g + "_ub"] = ll_ub
                else:
                    # TODO: we should check here that equations are well specified
                    d[g] = [str_expression(e) for e in eq_list]

            # if "controls_lb" not in d:
            #     for ind, g in enumerate(("controls_lb", "controls_ub")):
            #         eqs = []
            #         for i, eq in enumerate(d['arbitrage']):
            #             if "⟂" not in eq:
            #                 if ind == 0:
            #                     eq = "-inf"
            #                 else:
            #                     eq = "inf"
            #             else:
            #                 comp = eq.split("⟂")[1].strip()
            #                 v = self.symbols["controls"][i]
            #                 eq = decode_complementarity(comp, v+"[t]")[ind]
            #             eqs.append(eq)
            #         d[g] = eqs

            self.__equations__ = d

        return self.__equations__

    @property
    def definitions(self):

        from yaml import ScalarNode

        if self.__definitions__ is None:

            # at this stage, basic_symbols doesn't contain auxiliaries
            basic_symbols = self.symbols
            vars = sum(
                [basic_symbols[k] for k in basic_symbols.keys() if k != "parameters"],
                [],
            )

            # # auxiliaries = [remove_timing(parse_string(k)) for k in self.data.get('definitions', {})]
            # # auxiliaries = [str_expression(e) for e in auxiliaries]
            # # symbols['auxiliaries'] = auxiliaries

            if "definitions" not in self.data:
                self.__definitions__ = {}
                # self.__symbols__['auxiliaries'] = []

            elif isinstance(self.data["definitions"], ScalarNode):

                definitions = {}

                # new-style
                from lark import Token

                def_block_tree = parse_string(
                    self.data["definitions"], start="assignment_block"
                )
                def_block_tree = sanitize(
                    def_block_tree
                )  # just to replace (v,) by (v,0) # TODO: remove

                auxiliaries = []
                for eq_tree in def_block_tree.children:
                    lhs, rhs = eq_tree.children
                    tok_name: Token = lhs.children[0].children[0]
                    tok_date: Token = lhs.children[1].children[0]
                    name = tok_name.value
                    date = int(tok_date.value)
                    if name in vars:
                        raise Exception(
                            f"definitions:{tok_name.line}:{tok_name.column}: Auxiliary variable '{name}'' already defined."
                        )
                    if date != 0:
                        raise Exception(
                            f"definitions:{tok_name.line}:{tok_name.column}: Auxiliary variable '{name}' must be defined at date 't'."
                        )
                    # here we could check some stuff
                    from dolang import list_symbols

                    syms = list_symbols(rhs)
                    for p in syms.parameters:
                        if p in vars:
                            raise Exception(
                                f"definitions:{tok_name.line}: Symbol '{p}' is defined as a variable. Can't appear as a parameter."
                            )
                        if p not in self.symbols["parameters"]:
                            raise Exception(
                                f"definitions:{tok_name.line}: Paremeter '{p}' must be defined as a model symbol."
                            )
                    for v in syms.variables:
                        if v[0] not in vars:
                            raise Exception(
                                f"definitions:{tok_name.line}: Variable '{v[0]}[t]' is not defined."
                            )
                    auxiliaries.append(name)
                    vars.append(name)

                    definitions[str_expression(lhs)] = str_expression(rhs)

                self.__symbols__["auxiliaries"] = auxiliaries
                self.__definitions__ = definitions

            else:

                # old style
                from dolang.symbolic import remove_timing

                auxiliaries = [
                    remove_timing(parse_string(k))
                    for k in self.data.get("definitions", {})
                ]
                auxiliaries = [str_expression(e) for e in auxiliaries]
                self.__symbols__["auxiliaries"] = auxiliaries
                vars = self.variables
                auxs = []

                definitions = self.data["definitions"]
                d = dict()
                for i in range(len(definitions.value)):

                    kk = definitions.value[i][0]
                    if self.__compat__:
                        k = parse_string(kk.value)
                        if k.data == "symbol":
                            # TODO: warn that definitions should be timed
                            from dolang.grammar import create_variable

                            k = create_variable(k.children[0].value, 0)
                    else:
                        k = parse_string(kk.value, start="variable")
                    k = sanitize(k, variables=vars)

                    assert k.children[1].children[0].value == "0"

                    vv = definitions.value[i][1]
                    v = parse_string(vv)
                    v = sanitize(v, variables=vars)
                    v = str_expression(v)

                    key = str_expression(k)
                    vars.append(key)
                    d[key] = v
                    auxs.append(remove_timing(key))

                self.__symbols__["auxiliaries"] = auxs
                self.__definitions__ = d

        return self.__definitions__

    @property
    def name(self):
        try:
            return self.data["name"].value
        except Exception as e:
            return "Anonymous"

    @property
    def filename(self):
        try:
            return self.data["filename"].value
        except Exception as e:
            return "<string>"

    @property
    def infos(self):
        infos = {
            "name": self.name,
            "filename": self.filename,
            "type": "dtcc",
        }
        return infos

    @property
    def options(self):
        return self.data["options"]

    def get_calibration(self):

        # if self.__calibration__ is None:

        from dolang.symbolic import remove_timing

        import copy

        symbols = self.symbols
        calibration = dict()
        for k, v in self.data.get("calibration", {}).items():
            if v.tag == "tag:yaml.org,2002:str":

                expr = parse_string(v)
                expr = remove_timing(expr)
                expr = str_expression(expr)
            else:
                expr = float(v.value)
            kk = remove_timing(parse_string(k))
            kk = str_expression(kk)

            calibration[kk] = expr

        definitions = self.definitions

        initial_values = {
            "exogenous": 0,
            "expectations": 0,
            "values": 0,
            "controls": float("nan"),
            "states": float("nan"),
        }

        # variables defined by a model equation default to using these definitions
        initialized_from_model = {
            "values": "value",
            "expectations": "expectation",
            "direct_responses": "direct_response",
        }
        for k, v in definitions.items():
            kk = remove_timing(k)
            if kk not in calibration:
                if isinstance(v, str):
                    vv = remove_timing(v)
                else:
                    vv = v
                calibration[kk] = vv

        for symbol_group in symbols:
            if symbol_group not in initialized_from_model.keys():
                if symbol_group in initial_values:
                    default = initial_values[symbol_group]
                else:
                    default = float("nan")
                for s in symbols[symbol_group]:
                    if s not in calibration:
                        calibration[s] = default

        from dolang.triangular_solver import solve_triangular_system

        return solve_triangular_system(calibration)

    #     self.__calibration__ =  solve_triangular_system(calibration)

    # return self.__calibration__

    def get_domain(self):

        calibration = self.get_calibration()
        states = self.symbols["states"]

        sdomain = self.data.get("domain", {})
        for k in sdomain.keys():
            if k not in states:
                sdomain.pop(k)

        # backward compatibility
        if len(sdomain) == 0 and len(states) > 0:
            try:
                import warnings

                min = get_address(self.data, ["options:grid:a", "options:grid:min"])
                max = get_address(self.data, ["options:grid:b", "options:grid:max"])
                for i, s in enumerate(states):
                    sdomain[s] = [min[i], max[i]]
                # shall we raise a warning for deprecated syntax ?
            except Exception as e:
                pass

        if len(sdomain) == 0:
            return None

        if len(sdomain) < len(states):
            missing = [s for s in states if s not in sdomain]
            raise Exception(
                "Missing domain for states: {}.".format(str.join(", ", missing))
            )

        from dolo.compiler.objects import CartesianDomain
        from dolang.language import eval_data

        sdomain = eval_data(sdomain, calibration)

        domain = CartesianDomain(**sdomain)

        return domain

    def get_exogenous(self):

        if "exogenous" not in self.data:
            return {}

        exo = self.data["exogenous"]
        calibration = self.get_calibration()
        from dolang.language import eval_data

        exogenous = eval_data(exo, calibration)

        from dolo.numeric.processes import ProductProcess, Process

        if isinstance(exogenous, Process):
            # old style
            return exogenous
        elif isinstance(exo, list):
            # old style (2)
            return ProductProcess(*exogenous)
        else:
            # new style
            syms = self.symbols["exogenous"]
            # first we check that shocks are defined in the right order
            ssyms = []
            for k in exo.keys():
                vars = [v.strip() for v in k.split(",")]
                ssyms.append(vars)
            ssyms = tuple(sum(ssyms, []))
            if tuple(syms) != ssyms:
                from dolang.language import ModelError

                lc = exo.lc
                raise ModelError(
                    f"{lc.line}:{lc.col}: 'exogenous' section. Shocks specification must match declaration order. Found {ssyms}. Expected{tuple(syms)}"
                )

            return ProductProcess(*exogenous.values())

    @property
    def endo_grid(self):

        # determine bounds:
        domain = self.get_domain()
        min = domain.min
        max = domain.max

        options = self.data.get("options", {})

        # determine grid_type
        grid_type = get_type(options.get("grid"))
        if grid_type is None:
            grid_type = get_address(
                self.data, ["options:grid:type", "options:grid_type"]
            )
        if grid_type is None:
            raise Exception('Missing grid geometry ("options:grid:type")')

        args = {"min": min, "max": max}
        if grid_type.lower() in ("cartesian", "cartesiangrid"):
            from dolo.numeric.grids import UniformCartesianGrid

            orders = get_address(self.data, ["options:grid:n", "options:grid:orders"])
            if orders is None:
                orders = [20] * len(min)
            grid = UniformCartesianGrid(min=min, max=max, n=orders)
        elif grid_type.lower() in ("nonuniformcartesian", "nonuniformcartesiangrid"):
            from dolang.language import eval_data
            from dolo.numeric.grids import NonUniformCartesianGrid

            calibration = self.get_calibration()
            nodes = [eval_data(e, calibration) for e in self.data["options"]["grid"]]
            # each element of nodes should be a vector
            return NonUniformCartesianGrid(nodes)
        elif grid_type.lower() in ("smolyak", "smolyakgrid"):
            from dolo.numeric.grids import SmolyakGrid

            mu = get_address(self.data, ["options:grid:mu"])
            if mu is None:
                mu = 2
            grid = SmolyakGrid(min=min, max=max, mu=mu)
        else:
            raise Exception("Unknown grid type.")

        return grid


def get_type(d):
    try:
        s = d.tag
        return s.strip("!")
    except:
        v = d.get("type")
        return v


def get_address(data, address, default=None):

    if isinstance(address, list):
        found = [get_address(data, e, None) for e in address]
        found = [f for f in found if f is not None]
        if len(found) > 0:
            return found[0]
        else:
            return default
    fields = str.split(address, ":")
    while len(fields) > 0:
        data = data.get(fields[0])
        fields = fields[1:]
        if data is None:
            return default
    return eval_data(data)


import re

regex = re.compile("(.*)<=(.*)<=(.*)")


def decode_complementarity(comp, control):
    """
    # comp can be either:
    - None
    - "a<=expr" where a is a controls
    - "expr<=a" where a is a control
    - "expr1<=a<=expr2"
    """

    try:
        res = regex.match(comp).groups()
    except:
        raise Exception("Unable to parse complementarity condition '{}'".format(comp))

    res = [r.strip() for r in res]
    if res[1] != control:
        msg = "Complementarity condition '{}' incorrect. Expected {} instead of {}.".format(
            comp, control, res[1]
        )
        raise Exception(msg)

    return [res[0], res[2]]


class Model(SymbolicModel):
    """Model Object"""

    def __init__(self, data, check=True, compat=True):

        self.__compat__ = True

        super().__init__(data)

        self.model_type = "dtcc"
        self.__functions__ = None
        # self.__compile_functions__()
        self.set_changed(all="True")

        if check:
            self.symbols
            self.definitions
            self.calibration
            self.domain
            self.exogenous
            self.x_bounds
            self.functions

    def set_changed(self, all=False):
        self.__domain__ = None
        self.__exogenous__ = None
        self.__calibration__ = None
        if all:
            self.__symbols__ = None
            self.__definitions__ = None
            self.__variables__ = None
            self.__equations__ = None

    def set_calibration(self, *pargs, **kwargs):
        if len(pargs) == 1:
            self.set_calibration(**pargs[0])
        self.set_changed()
        self.data["calibration"].update(kwargs)

    @property
    def calibration(self):
        if self.__calibration__ is None:
            calibration_dict = super().get_calibration()
            from dolo.compiler.misc import CalibrationDict, calibration_to_vector

            calib = calibration_to_vector(self.symbols, calibration_dict)
            self.__calibration__ = CalibrationDict(self.symbols, calib)  #
        return self.__calibration__

    @property
    def exogenous(self):
        if self.__exogenous__ is None:
            self.__exogenous__ = super(self.__class__, self).get_exogenous()
        return self.__exogenous__

    @property
    def domain(self):
        if self.__domain__ is None:
            self.__domain__ = super().get_domain()
        return self.__domain__

    def discretize(self, grid_options=None, dprocess_options={}):

        dprocess = self.exogenous.discretize(**dprocess_options)

        if grid_options is None:
            endo_grid = self.endo_grid
        else:
            endo_grid = self.domain.discretize(**grid_options)

        from dolo.numeric.grids import ProductGrid

        grid = ProductGrid(dprocess.grid, endo_grid, names=["exo", "endo"])
        return [grid, dprocess]

    def __compile_functions__(self):

        from dolang.function_compiler import make_method_from_factory

        from dolang.vectorize import standard_function
        from dolo.compiler.factories import get_factory
        from .misc import LoosyDict

        equivalent_function_names = {
            "equilibrium": "arbitrage",
            "optimality": "arbitrage",
        }
        functions = LoosyDict(equivalences=equivalent_function_names)
        original_functions = {}
        original_gufunctions = {}

        funnames = [*self.equations.keys()]
        if len(self.definitions) > 0:
            funnames = funnames + ["auxiliary"]

        import dolo.config

        debug = dolo.config.debug

        for funname in funnames:

            fff = get_factory(self, funname)
            fun, gufun = make_method_from_factory(fff, vectorize=True, debug=debug)
            n_output = len(fff.content)
            functions[funname] = standard_function(gufun, n_output)
            original_gufunctions[funname] = gufun  # basic gufun function
            original_functions[funname] = fun  # basic numba fun

        self.__original_functions__ = original_functions
        self.__original_gufunctions__ = original_gufunctions
        self.__functions__ = functions

    @property
    def functions(self):
        if self.__functions__ is None:
            self.__compile_functions__()
        return self.__functions__

    def __str__(self):

        from dolo.misc.termcolor import colored
        from numpy import zeros

        s = """
        Model:
        ------
        name: "{name}"
        type: "{type}"
        file: "{filename}\n""".format(
            **self.infos
        )

        ss = "\nEquations:\n----------\n\n"
        res = self.residuals()
        res.update({"definitions": zeros(1)})

        equations = self.equations.copy()
        definitions = self.definitions
        tmp = []
        for deftype in definitions:
            tmp.append(deftype + " = " + definitions[deftype])
        definitions = {"definitions": tmp}
        equations.update(definitions)
        # for eqgroup, eqlist in self.symbolic.equations.items():
        for eqgroup in res.keys():
            if eqgroup == "auxiliary":
                continue
            if eqgroup == "definitions":
                eqlist = equations[eqgroup]
                # Update the residuals section with the right number of empty
                # values. Note: adding 'zeros' was easiest (rather than empty
                # cells), since other variable types have  arrays of zeros.
                res.update({"definitions": [None for i in range(len(eqlist))]})
            else:
                eqlist = equations[eqgroup]
            ss += "{}\n".format(eqgroup)
            for i, eq in enumerate(eqlist):
                val = res[eqgroup][i]
                if val is None:
                    ss += " {eqn:2} : {eqs}\n".format(eqn=str(i + 1), eqs=eq)
                else:
                    if abs(val) < 1e-8:
                        val = 0
                    vals = "{:.4f}".format(val)
                    if abs(val) > 1e-8:
                        vals = colored(vals, "red")
                    ss += " {eqn:2} : {vals} : {eqs}\n".format(
                        eqn=str(i + 1), vals=vals, eqs=eq
                    )
            ss += "\n"
        s += ss

        return s

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):

        from dolang.latex import eq2tex

        # general informations
        infos = self.infos
        table_infos = """
        <table>
         <td><b>Model</b></td>
        <tr>
        <td>name</td>
        <td>{name}</td>
        </tr>
        <tr>
        <td>type</td>
        <td>{type}</td>
        </tr>
        <tr>
        <td>filename</td>
        <td>{filename}</td>
        </tr>
        </table>""".format(
            name=infos["name"],
            type=infos["type"],
            filename=infos["filename"].replace("<", "&lt").replace(">", "&gt"),
        )

        # Equations and residuals
        resids = self.residuals()
        equations = self.equations.copy()
        # Create definitions equations and append to equations dictionary
        definitions = self.definitions
        tmp = []
        for deftype in definitions:
            tmp.append(deftype + " = " + definitions[deftype])

        definitions = {"definitions": tmp}
        equations.update(definitions)

        variables = sum([e for k, e in self.symbols.items() if k != "parameters"], [])
        table = '<tr><td><b>Type</b></td><td style="width:80%"><b>Equation</b></td><td><b>Residual</b></td></tr>\n'

        for eq_type in equations:

            eq_lines = []
            for i in range(len(equations[eq_type])):
                eq = equations[eq_type][i]
                # if eq_type in ('expectation','direct_response'):
                #     vals = ''
                if eq_type not in ("arbitrage", "transition", "arbitrage_exp"):
                    vals = ""
                else:
                    val = resids[eq_type][i]
                    if abs(val) > 1e-8:
                        vals = '<span style="color: red;">{:.4f}</span>'.format(val)
                    else:
                        vals = "{:.3f}".format(val)
                if "⟂" in eq:
                    # keep only lhs for now
                    eq, comp = str.split(eq, "⟂")
                if "|" in eq:
                    # keep only lhs for now
                    eq, comp = str.split(eq, "|")
                lat = eq2tex(variables, eq)
                lat = "${}$".format(lat)
                line = [lat, vals]
                h = eq_type if i == 0 else ""
                fmt_line = "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(h, *line)
                #         print(fmt_line)
                eq_lines.append(fmt_line)
            table += str.join("\n", eq_lines)
        table = "<table>{}</table>".format(table)

        return table_infos + table

    @property
    def x_bounds(self):

        if "controls_ub" in self.functions:
            fun_lb = self.functions["controls_lb"]
            fun_ub = self.functions["controls_ub"]
            return [fun_lb, fun_ub]
        elif "arbitrage_ub" in self.functions:
            fun_lb = self.functions["arbitrage_lb"]
            fun_ub = self.functions["arbitrage_ub"]
            return [fun_lb, fun_ub]
        else:
            return None

    def residuals(self, calib=None):

        from dolo.algos.steady_state import residuals

        return residuals(self, calib)

    def eval_formula(self, expr, dataframe=None, calib=None):

        from dolo.compiler.eval_formula import eval_formula

        if calib is None:
            calib = self.calibration
        return eval_formula(expr, dataframe=dataframe, context=calib)

```

File: dolo/algos/time_iteration.py
```py
"""Time Iteration Algorithm"""

import numpy  # For numerical computations
from dolo import dprint  # For debug printing
from dolo.compiler.model import Model  # Base model class
from dolo.numeric.processes import DiscretizedIIDProcess  # For discretizing shocks
from dolo.numeric.decision_rule import DecisionRule  # For policy functions
from dolo.numeric.grids import CartesianGrid  # For state space discretization


def residuals_simple(f, g, s, x, dr, dprocess, parms):  # Compute arbitrage equation residuals

    N = s.shape[0]  # Number of grid points
    n_s = s.shape[1]  # Number of state variables

    res = numpy.zeros_like(x)  # Initialize residuals array

    for i_ms in range(dprocess.n_nodes):  # Loop over exogenous states

        # solving on grid for markov index i_ms
        m = numpy.tile(dprocess.node(i_ms), (N, 1))  # Current exogenous state
        xm = x[i_ms, :, :]  # Current controls

        for I_ms in range(dprocess.n_inodes(i_ms)):  # Loop over future states
            M = numpy.tile(dprocess.inode(i_ms, I_ms), (N, 1))  # Next period exogenous
            prob = dprocess.iweight(i_ms, I_ms)  # Transition probability
            S = g(m, s, xm, M, parms)  # Next period states
            XM = dr.eval_ijs(i_ms, I_ms, S)  # Next period controls
            rr = f(m, s, xm, M, S, XM, parms)  # Arbitrage equation residuals
            res[i_ms, :, :] += prob * rr  # Add weighted residuals

    return res  # Return total residuals


from .results import TimeIterationResult, AlgoResult  # For returning solution results
from dolo.misc.itprinter import IterationsPrinter  # For iteration output
import copy  # For deep copying objects


def time_iteration(
    model: Model,  # Model to solve
    *,  #
    dr0: DecisionRule = None,  # Initial guess for decision rule
    verbose: bool = True,  # Whether to print progress
    details: bool = True,  # Whether to return detailed results
    ignore_constraints: bool = False,  # Whether to ignore bounds
    trace: bool = False,  # Whether to store iteration history
    dprocess=None,  # Optional custom discretization
    maxit=1000,  # Maximum iterations
    inner_maxit=10,  # Maximum iterations for inner solver
    tol=1e-6,  # Convergence tolerance
    hook=None,  # Optional callback function
    interp_method="cubic",  # Interpolation method
    # obsolete
    with_complementarities=None,  # Deprecated option
) -> TimeIterationResult:
    """Finds a global solution for ``model`` using backward time-iteration.


    This algorithm iterates on the residuals of the arbitrage equations

    Parameters
    ----------
    model : Model
        model to be solved
    verbose : bool
        if True, display iterations
    dr0 : decision rule
        initial guess for the decision rule
    with_complementarities : bool (True)
        if False, complementarity conditions are ignored
    maxit: maximum number of iterations
    inner_maxit: maximum number of iteration for inner solver
    tol: tolerance criterium for successive approximations
    hook: Callable
        function to be called within each iteration, useful for debugging purposes


    Returns
    -------
    decision rule :
        approximated solution
    """

    # deal with obsolete options
    if with_complementarities is not None:
        # TODO warn
        pass
    else:
        with_complementarities = not ignore_constraints  # Set complementarity flag based on constraints

    if trace:  # Initialize trace storage if requested
        trace_details = []
    else:
        trace_details = None

    from dolo import dprint  # Import debug printing utility

    def vprint(t):  # Helper function for verbose output
        if verbose:
            print(t)

    grid, dprocess_ = model.discretize()  # Get discretized state space

    if dprocess is None:  # Use default discretization if none provided
        dprocess = dprocess_

    n_ms = dprocess.n_nodes  # Number of exogenous states
    n_mv = dprocess.n_inodes(0)  # Number of integration nodes

    x0 = model.calibration["controls"]  # Get initial control values
    parms = model.calibration["parameters"]  # Get model parameters
    n_x = len(x0)  # Number of control variables
    n_s = len(model.symbols["states"])  # Number of state variables

    endo_grid = grid["endo"]  # Grid for endogenous states
    exo_grid = grid["exo"]  # Grid for exogenous states

    mdr = DecisionRule(  # Create decision rule object
        exo_grid, endo_grid, dprocess=dprocess, interp_method=interp_method
    )

    s = mdr.endo_grid.nodes  # Get grid nodes
    N = s.shape[0]  # Number of grid points

    controls_0 = numpy.zeros((n_ms, N, n_x))  # Initialize control values
    if dr0 is None:  # If no initial guess provided
        controls_0[:, :, :] = x0[None, None, :]  # Use calibrated controls
    else:
        if isinstance(dr0, AlgoResult):  # Extract decision rule if needed
            dr0 = dr0.dr
        try:
            for i_m in range(n_ms):  # Try evaluating on grid
                controls_0[i_m, :, :] = dr0(i_m, s)
        except Exception:
            for i_m in range(n_ms):  # Fall back to direct evaluation
                m = dprocess.node(i_m)  # Get exogenous state
                controls_0[i_m, :, :] = dr0(m, s)  # Evaluate initial guess

    f = model.functions["arbitrage"]  # Get arbitrage equations
    g = model.functions["transition"]  # Get transition equations

    if "arbitrage_lb" in model.functions and with_complementarities == True:  # Handle bounds
        lb_fun = model.functions["arbitrage_lb"]  # Lower bound function
        ub_fun = model.functions["arbitrage_ub"]  # Upper bound function
        lb = numpy.zeros_like(controls_0) * numpy.nan  # Initialize lower bounds
        ub = numpy.zeros_like(controls_0) * numpy.nan  # Initialize upper bounds
        for i_m in range(n_ms):  # Compute bounds at each point
            m = dprocess.node(i_m)[None, :]  # Get exogenous state
            p = parms[None, :]  # Get parameters
            m = numpy.repeat(m, N, axis=0)  # Repeat for each grid point
            p = numpy.repeat(p, N, axis=0)  # Repeat parameters

            lb[i_m, :, :] = lb_fun(m, s, p)  # Evaluate lower bounds
            ub[i_m, :, :] = ub_fun(m, s, p)  # Evaluate upper bounds

    else:
        with_complementarities = False  # Disable bounds if not provided

    sh_c = controls_0.shape  # Store shape for reshaping

    controls_0 = controls_0.reshape((-1, n_x))  # Flatten controls for solver

    from dolo.numeric.optimize.newton import newton, SerialDifferentiableFunction  # Import solvers
    from dolo.numeric.optimize.ncpsolve import ncpsolve  # Import complementarity solver

    err = 10  # Initialize error measure
    it = 0  # Initialize iteration counter

    if with_complementarities:  # If using bounds
        lb = lb.reshape((-1, n_x))  # Reshape bounds to match controls
        ub = ub.reshape((-1, n_x))  # Reshape bounds to match controls

    itprint = IterationsPrinter(  # Setup iteration printer
        ("N", int),  # Iteration number
        ("Error", float),  # Current error
        ("Gain", float),  # Error reduction
        ("Time", float),  # Iteration time
        ("nit", int),  # Inner iterations
        verbose=verbose,
    )
    itprint.print_header("Start Time Iterations.")  # Print header

    import time  # For timing iterations

    t1 = time.time()  # Start timing

    err_0 = numpy.nan  # Initialize previous error
    verbit = verbose == "full"  # Set verbosity level

    while err > tol and it < maxit:  # Main iteration loop

        it += 1  # Increment counter

        t_start = time.time()  # Start iteration timer

        mdr.set_values(controls_0.reshape(sh_c))  # Update decision rule

        if trace:  # Store iteration history if requested
            trace_details.append({"dr": copy.deepcopy(mdr)})

        fn = lambda x: residuals_simple(  # Define residual function
            f, g, s, x.reshape(sh_c), mdr, dprocess, parms
        ).reshape((-1, n_x))
        dfn = SerialDifferentiableFunction(fn)  # Make function differentiable

        res = fn(controls_0)  # Compute residuals

        if hook:  # Call hook if provided
            hook()

        if with_complementarities:  # Solve with bounds if needed
            [controls, nit] = ncpsolve(
                dfn, lb, ub, controls_0, verbose=verbit, maxit=inner_maxit
            )
        else:  # Solve without bounds
            [controls, nit] = newton(dfn, controls_0, verbose=verbit, maxit=inner_maxit)

        err = abs(controls - controls_0).max()  # Compute error

        err_SA = err / err_0  # Compute error reduction
        err_0 = err  # Store error for next iteration

        controls_0 = controls  # Update controls

        t_finish = time.time()  # End iteration timer
        elapsed = t_finish - t_start  # Compute iteration time

        itprint.print_iteration(N=it, Error=err_0, Gain=err_SA, Time=elapsed, nit=nit)  # Print progress

    controls_0 = controls.reshape(sh_c)  # Reshape solution

    mdr.set_values(controls_0)  # Set final values
    if trace:  # Store final iteration if tracing
        trace_details.append({"dr": copy.deepcopy(mdr)})

    itprint.print_finished()  # Print completion message

    if not details:  # Return just decision rule if no details wanted
        return mdr

    return TimeIterationResult(  # Return full results
        mdr,  # Decision rule
        it,  # Iterations
        with_complementarities,  # Whether bounds were used
        dprocess,  # Discretization process
        err < tol,  # Whether converged
        tol,  # Tolerance used
        err,  # Final error
        None,  # Log (not used)
        trace_details,  # Iteration history
    )

```

File: dolo/compiler/recipes.py
```py
"""
ideas :
-  recursive blocks           [by default]
- (order left hand side ?)    [by default]
- dependency across blocks
- dummy blocks that are basically substituted everywhere else
"""

import os, yaml, sys  # Core modules for file/system operations and YAML parsing

if getattr(sys, "frozen", False):  # Check if running as PyInstaller executable
    # we are running in a |PyInstaller| bundle
    DIR_PATH = sys._MEIPASS  # Get bundle directory path
else:
    DIR_PATH, this_filename = os.path.split(__file__)  # Get module directory path

DATA_PATH = os.path.join(DIR_PATH, "recipes.yaml")  # Path to model recipe definitions

with open(DATA_PATH, "rt", encoding="utf-8") as f:  # Load recipe file with UTF-8 encoding
    recipes = yaml.safe_load(f)  # Parse YAML into recipe dictionary

```

File: dolo/numeric/matrix_equations.py
```py
from dolo.numeric.tensor import sdot, mdot  # For matrix operations in perturbation methods

import numpy as np  # For numerical array operations

TOL = 1e-10  # Tolerance for numerical convergence


# credits : second_order_solver is adapted from Sven Schreiber's port of Uhlig's Toolkit.
def second_order_solver(FF, GG, HH, eigmax=1.0 + 1e-6):  # Solves second-order perturbation equations

    # from scipy.linalg import qz
    from dolo.numeric.extern.qz import qzordered  # For generalized Schur decomposition

    from numpy import (
        array,
        mat,
        c_,
        r_,
        eye,
        zeros,
        real_if_close,
        diag,
        allclose,
        where,
        diagflat,
    )  # Import numpy functions for matrix operations
    from numpy.linalg import solve  # For solving linear systems

    Psi_mat = array(FF)  # Convert first-order derivatives to array
    Gamma_mat = array(-GG)  # Negative transition matrix
    Theta_mat = array(-HH)  # Second-order derivatives
    m_states = FF.shape[0]  # Number of state variables

    Xi_mat = r_[
        c_[Gamma_mat, Theta_mat], c_[eye(m_states), zeros((m_states, m_states))]
    ]  # Build augmented matrix for QZ decomposition

    Delta_mat = r_[
        c_[Psi_mat, zeros((m_states, m_states))],
        c_[zeros((m_states, m_states)), eye(m_states)],
    ]  # Build companion matrix

    [Delta_up, Xi_up, UUU, VVV, eigval] = qzordered(
        Delta_mat,
        Xi_mat,
    )  # Compute ordered QZ decomposition

    VVVH = VVV.T  # Transpose of right eigenvectors
    VVV_2_1 = VVVH[m_states : 2 * m_states, :m_states]  # Extract top-right block
    VVV_2_2 = VVVH[m_states : 2 * m_states, m_states : 2 * m_states]  # Extract bottom-right block
    UUU_2_1 = UUU[m_states : 2 * m_states, :m_states]  # Extract relevant block of U matrix
    PP = -solve(VVV_2_1, VVV_2_2)  # Compute policy function matrix

    # slightly different check than in the original toolkit:
    assert allclose(real_if_close(PP), PP.real)  # Verify solution is real-valued
    PP = PP.real  # Extract real part

    return [eigval, PP]  # Return eigenvalues and policy matrix


def solve_sylvester_vectorized(*args):  # Solve vectorized Sylvester equation
    from numpy import kron  # For Kronecker product operations
    from numpy.linalg import solve  # For linear system solution

    vec = lambda M: M.ravel()  # Helper to flatten matrix to vector
    n = args[0][0].shape[0]  # Get first dimension
    q = args[0][1].shape[0]  # Get second dimension
    K = vec(args[-1])  # Vectorize right-hand side
    L = sum([kron(A, B.T) for (A, B) in args[:-1]])  # Build coefficient matrix
    X = solve(L, -K)  # Solve linear system
    return X.reshape((n, q))  # Reshape solution to matrix form


def solve_sylvester(A, B, C, D, Ainv=None, method="linear"):  # Solve generalized Sylvester equation
    # Solves equation : A X + B X [C,...,C] + D = 0
    # where X is a multilinear function whose dimension is determined by D
    # inverse of A can be optionally specified as an argument

    n_d = D.ndim - 1  # Get number of dimensions
    n_v = C.shape[1]  # Get number of variables

    n_c = D.size // n_v**n_d  # Calculate size of control space

    DD = D.reshape(n_c, n_v**n_d)  # Reshape D matrix

    if n_d == 1:  # Handle 1D case
        CC = C
    else:  # Handle higher dimensional cases
        CC = np.kron(C, C)  # Compute Kronecker product
    for i in range(n_d - 2):  # Additional Kronecker products if needed
        CC = np.kron(CC, C)

    if method == "linear":  # Linear solution method
        I = np.eye(CC.shape[0])  # Identity matrix
        XX = solve_sylvester_vectorized((A, I), (B, CC), DD)  # Solve using vectorized method

    else:  # Use slycot solver
        # we use slycot by default
        import slycot  # Import specialized solver

        if Ainv != None:  # Use provided inverse if available
            Q = sdot(Ainv, B)  # Compute Q matrix
            S = sdot(Ainv, DD)  # Compute S matrix
        else:  # Compute inverse
            Q = np.linalg.solve(A, B)  # Solve for Q
            S = np.linalg.solve(A, DD)  # Solve for S

        n = n_c  # Number of equations
        m = n_v**n_d  # Size of solution

        XX = slycot.sb04qd(n, m, Q, CC, -S)  # Solve using slycot

    X = XX.reshape((n_c,) + (n_v,) * (n_d))  # Reshape solution

    return X  # Return solution matrix


class BKError(Exception):  # Custom exception for Blanchard-Kahn errors
    def __init__(self, type):
        self.type = type

    def __str__(self):
        return "Blanchard-Kahn error ({0})".format(self.type)  # Format error message

```

File: dolo/compiler/recipes.yaml
```yaml
dtcc:

    model_spec: dtcc

    symbols: ['exogenous', 'states', 'controls', 'poststates', 'rewards', 'values', 'expectations', 'shocks', 'parameters']

    specs:

        arbitrage:

            eqs:

                - ['exogenous',0,'m']
                - ['states',0,'s']
                - ['controls',0,'x']

                - ['exogenous',1,'M']
                - ['states',1,'S']
                - ['controls',1,'X']

                - ['parameters',0,'p']

            complementarities:

                left-right:

                    - ['exogenous',0,'m']
                    - ['states', 0, 's']
                    - ['parameters', 0, 'p']

                middle: ['controls', 0, 's']


        transition:

            target: ['states',0,'S']

            eqs:
                - ['exogenous',-1,'m']
                - ['states',-1,'s']
                - ['controls',-1,'x']
                - ['exogenous',0,'M']
                - ['parameters', 0, 'p']

        value:

            optional: True

            target: ['values',0,'v']
            recursive: False

            eqs:
                - ['exogenous',0,'m']
                - ['states',0,'s']
                - ['controls',0,'x']
                - ['values', 0, 'v']

                - ['exogenous',1,'M']
                - ['states',1,'S']
                - ['controls',1,'X']
                - ['values', 1, 'V']

                - ['parameters',0,'p']

        expectation:
            optional: True
            target: ['expectations',0,'z']
            recursive: False
            eqs:
                - ['exogenous',1,'M']
                - ['states',1,'S']
                - ['controls',1,'X']
                - ['parameters',0,'p']

        half_transition:
            optional: True
            target: ['states', 0, 'S']
            recursive: False
            eqs:
                - ['exogenous', -1, 'm']
                - ['poststates', -1, 'a']
                - ['exogenous', 0, 'M']
                - ['parameters', 0, 'p']

        direct_response_egm:

            optional: True
            recursive: True
            target: ['controls', 0,'x']

            eqs:
                - ['exogenous',0,'m']
                - ['poststates',0,'a']
                - ['expectations', 0, 'z']
                - ['parameters',0,'p']

        reverse_state:

            optional: True
            recursive: True
            target: ['states', 0, 's']
            eqs:
                - ['exogenous', 0, 'm']
                - ['poststates', 0, 'a']
                - ['controls', 0, 'x']
                - ['parameters',0,'p']


            

        direct_response:

            optional: True
            recursive: True
            target: ['controls', 0,'x']

            eqs:
                - ['exogenous',0,'m']
                - ['states',0,'s']
                - ['expectations', 0, 'z']
                - ['parameters',0,'p']

        felicity:

            optional: True
            recursive: True

            target: ['rewards', 0,'r']

            eqs:
                - ['exogenous',0,'m']
                - ['states',0,'s']
                - ['controls',0,'x']
                - ['parameters', 0, 'p']


        arbitrage_exp:

            optional: True

            eqs:
                - ['states',0,'s']
                - ['controls',0,'x']
                - ['parameters', 0, 'p']

```

File: dolo/compiler/objects.py
```py
from dolo.numeric.distribution import *

#
from dataclasses import dataclass
from dolang.language import language_element

# not sure we'll keep that
import numpy as np
from typing import List, Union

Scalar = Union[int, float]

# not really a language element though
# @language_element


class Domain:
    pass


class CartesianDomain(Domain, dict):
    def __init__(self, **kwargs):
        super().__init__()
        for k, w in kwargs.items():
            v = kwargs[k]
            self[k] = np.array(v, dtype=float)

    def discretize(self, n=None):
        if n == None:
            n = [10] * (len(self.min))
        from dolo.numeric.grids import UniformCartesianGrid

        return UniformCartesianGrid(self.min, self.max, n)

    @property
    def states(self):
        l = tuple(self.keys())
        return l

    @property
    def min(self):
        return np.array([self[e][0] for e in self.states])

    @property
    def max(self):
        return np.array([self[e][1] for e in self.states])


# these are dummy objects so far
#
# @language_element
# @dataclass
# class UNormal:
#     mu: float
#     sigma: float
#     signature = {'mu': 'float', 'sigma': 'float'}
#


# @language_element
# @dataclass
# class MvNormal:
#     Mu: List[float]
#     Sigma: List[List[float]]
#     signature = {'Mu': 'list(float)', 'Sigma': 'Matrix'}


# %%


@language_element
class Conditional:

    signature = {"condition": None, "type": None, "arguments": None}

    def __init__(self, condition, type, arguments):
        self.condition = condition
        self.type = type
        self.arguments = arguments


@language_element
class Product:
    def __init__(self, *args: List):
        self.factors = args


@language_element
def Matrix(*lines):
    mat = np.array(lines, np.float64)
    assert mat.ndim == 2
    return mat


@language_element
def Vector(*elements):
    mat = np.array(elements, np.float64)
    assert mat.ndim == 1
    return mat

```

File: dolo/compiler/factories.py
```py
# plainformatter = get_ipython().display_formatter.formatters['text/plain']
# del plainformatter.type_printers[dict]

import yaml
import numpy as np
from typing import List

import ast
from ast import BinOp, Sub

from typing import Dict

import dolang
from dolang.grammar import str_expression
from dolang.symbolic import parse_string
from dolang.symbolic import time_shift
from dolang.symbolic import Sanitizer
from dolang.factory import FlatFunctionFactory


def get_name(e):
    return ast.parse(e).body[0].value.func.id


def reorder_preamble(pr):

    from dolang.triangular_solver import triangular_solver, get_incidence

    inc = get_incidence(pr)
    order = triangular_solver(inc)
    d = dict()
    prl = [*pr.items()]
    for o in order:
        k, v = prl[o]
        d[k] = v
    return d


def shift_spec(specs, tshift):
    ss = dict()
    if "target" in specs:
        e = specs["target"]
        ss["target"] = [e[0], e[1] + tshift, e[2]]
    ss["eqs"] = [
        ([e[0], e[1] + tshift, e[2]] if e[0] != "parameters" else e)
        for e in specs["eqs"]
    ]
    return ss


def get_factory(model, eq_type: str, tshift: int = 0):

    from dolo.compiler.model import decode_complementarity

    from dolo.compiler.recipes import recipes
    from dolang.symbolic import stringify, stringify_symbol

    equations = model.equations

    if eq_type == "auxiliary":
        eqs = ["{}".format(s) for s in model.symbols["auxiliaries"]]
        specs = {
            "eqs": [
                ["exogenous", 0, "m"],
                ["states", 0, "s"],
                ["controls", 0, "x"],
                ["parameters", 0, "p"],
            ]
        }
    else:
        eqs = equations[eq_type]
        if eq_type in ("arbitrage_lb", "arbitrage_ub"):
            specs = {
                "eqs": recipes["dtcc"]["specs"]["arbitrage"]["complementarities"][
                    "left-right"
                ]
            }
        else:
            specs = recipes["dtcc"]["specs"][eq_type]

    specs = shift_spec(specs, tshift=tshift)

    preamble_tshift = set([s[1] for s in specs["eqs"] if s[0] == "states"])
    preamble_tshift = preamble_tshift.intersection(
        set([s[1] for s in specs["eqs"] if s[0] == "controls"])
    )

    args = []
    for sg in specs["eqs"]:
        if sg[0] == "parameters":
            args.append([s for s in model.symbols["parameters"]])
        else:
            args.append([(s, sg[1]) for s in model.symbols[sg[0]]])
    args = [[stringify_symbol(e) for e in vg] for vg in args]

    arguments = dict(zip([sg[2] for sg in specs["eqs"]], args))

    # temp
    eqs = [eq.split("⟂")[0].strip() for eq in eqs]

    if "target" in specs:
        sg = specs["target"]
        targets = [(s, sg[1]) for s in model.symbols[sg[0]]]
        eqs = [eq.split("=")[1] for eq in eqs]
    else:
        eqs = [
            ("({1})-({0})".format(*eq.split("=")) if "=" in eq else eq) for eq in eqs
        ]
        targets = [("out{}".format(i), 0) for i in range(len(eqs))]

    eqs = [str.strip(eq) for eq in eqs]

    eqs = [dolang.parse_string(eq) for eq in eqs]
    es = Sanitizer(variables=model.variables)
    eqs = [es.transform(eq) for eq in eqs]

    eqs = [time_shift(eq, tshift) for eq in eqs]

    eqs = [stringify(eq) for eq in eqs]

    eqs = [str_expression(eq) for eq in eqs]

    targets = [stringify_symbol(e) for e in targets]

    # sanitize defs ( should be )
    defs = dict()

    for k in model.definitions:
        val = model.definitions[k]
        # val = es.transform(dolang.parse_string(val))
        for t in preamble_tshift:
            s = stringify(time_shift(k, t))
            if isinstance(val, str):
                vv = stringify(time_shift(val, t))
            else:
                vv = str(val)
            defs[s] = vv

    preamble = reorder_preamble(defs)

    eqs = dict(zip(targets, eqs))
    ff = FlatFunctionFactory(preamble, eqs, arguments, eq_type)

    return ff

```
</file_contents>

<meta prompt 1 = "dolo_add-docs-to-py-files_egm-only">
You are an experienced software engineer who has been tasked with adding documentation to the python code in the ‘dolo’ project.

PHASE 1:
Begin by thoroughly understanding the structure of the dolo package, with a particular emphasis on understanding the relationship between the documentation found in the dolo/docs directory about how to use the package, and the implementation in the dolo directory and the examples, with particular focus on the examples in examples/notebooks_docs-added.

Using that understanding, add inline comments (that is, comments that do not change the number of lines in the file) to all python files in the subdirectories of dolo/algos, ignoring only files named `__init__.py`.  In this phase, do not make ANY changes that would change the number of lines in the file (like, do not add or delete whitespace lines).  If there is already a comment, its content should be preserved exactly as-is; do not add anything to already-commented lines.

Process for commenting each file:
1. First read and understand the entire file thoroughly
   - Make sure you understand how it relates to the documentation in dolo/docs
   - Make sure you understand the purpose of each function
   - Make sure you understand the meaning of each non-trivial line of code
2. Identify all lines requiring comments
3. Plan appropriate comments that explain:
   - The purpose of each function
   - The relationship to concepts in the documentation
   - The meaning of each non-trivial line of code
4. Only then begin adding comments, and complete the entire file
5. Verify that no uncommented lines remain by:
   a. Checking every import statement has a comment
   b. Checking every variable assignment has a comment
   c. Checking every function definition has a comment
   d. Checking every return statement has a comment
   e. Checking every operation has a comment
6. Only proceed to the next file after confirming ALL lines have appropriate comments


Important rules for adding comments:
1. Do not change the number of lines in any file
2. Do not modify or delete ANY existing comments
3. Only add comments to currently uncommented lines
4. Do not add or remove whitespace lines
5. Every function must have at least one comment explaining its purpose
6. Every import statement should have a comment explaining what it's used for
7. Every non-trivial line of code should have an inline comment explaining what it does


NOT ALLOWED:
1. Replacing or modifying existing comment
2. Adding a comment to a line that does not already have a comment


Examples:

The first line of 'commands.py' is already commented: "# this is for compatibility purposes only"
- You should NOT modify this comment in any way.

The second line of 'commands.py' is blank
- You may add a comment to this line


PHASE 2:
After you have added the inline comments to each file, you can add docstrings for each function in the file that does not already have one.  The docstrings should make explicit reference to the documentation and to other relevant parts of the codebase.  If there are instances of the function being used in any file anywhere in directory tree examples/models/ or models_/, make an excerpt from the example that illustrates the usage as found in that file.

The goal is for a reader of the code to be able to understand:
1. How (if at all) the code line is related to high-level concepts in the documentation (like ‘state variables’)
2. How (if at all) the code line uses tools from other .py files (if there is space, list the .py file used)


</meta prompt 1>
<user_instructions>
Read and apply the instructions in AIPrompts/dolo_add-docs-to-py-files_egm-only.md, but apply them only to the file dolo/algos/egm.py.  Leave all other files unchanged.


</user_instructions>
By following these refined instructions, you will enhance the `egm.py` file with
thorough documentation while preserving behavior and structure **exactly**. This
ensures the solution remains functionally identical, with only
documentation added.
