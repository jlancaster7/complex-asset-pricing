# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell
```

### Running Tests
```bash
# Run all tests with proper Python path
export PYTHONPATH=$PWD/src:$PYTHONPATH && poetry run pytest tests/ -v

# Run specific test file
export PYTHONPATH=$PWD/src:$PYTHONPATH && poetry run pytest tests/test_yield_curve.py -v

# Run with coverage
export PYTHONPATH=$PWD/src:$PYTHONPATH && poetry run pytest --cov=ls_pricing tests/
```

### Code Quality
```bash
# Format code with Black
poetry run black src/ tests/

# Sort imports
poetry run isort src/ tests/

# Type checking
poetry run mypy src/
```

## Architecture Overview

This codebase implements the Longstaff-Schwartz method for pricing American options using Hull-White interest rate modeling. The architecture follows a clear separation of concerns:

### Core Models (`src/ls_pricing/core/`)
- **YieldCurve**: Handles interest rate curve interpolation, discount factors, and forward rates. Uses cubic spline interpolation with extrapolation.
- **HullWhiteModel**: Implements the one-factor Hull-White model `dr(t) = [θ(t) - a·r(t)]dt + σ·dW(t)`. Provides short rate simulation and analytical zero bond pricing.

### Pricing Engines (`src/ls_pricing/engine/`)
- **MonteCarloEngine**: Manages Monte Carlo simulations for path generation and European option pricing. Designed to be extended for the Longstaff-Schwartz algorithm.

### Key Design Principles
1. **Type Safety**: Extensive use of type hints throughout
2. **Validation**: Input validation in constructors with clear error messages
3. **Flexibility**: Methods accept both scalar floats and numpy arrays where appropriate
4. **Testability**: Comprehensive test suite with property-based tests

## Implementation Status

### Completed (Phase 1)
- YieldCurve class with interpolation and rate calculations
- HullWhiteModel with Euler discretization for rate simulation
- Basic MonteCarloEngine for path generation
- Comprehensive test suite

### Pending Implementation
- Longstaff-Schwartz algorithm (Phase 3)
- Regression basis functions (Laguerre polynomials)
- American option pricing logic
- Hull-White calibration to swaptions
- Numba optimizations for performance

## Important Implementation Notes

1. **Follow KISS Principle**: The docs emphasize keeping implementations simple. Always choose the simpler approach when the technical plan offers options.

2. **Euler Discretization**: The Hull-White simulation uses Euler scheme, not exact simulation, as per the simplification guidelines.

3. **Regression Choices**: Start with simple OLS regression and Laguerre polynomials with degree 3.

4. **Data Files**: Market data is stored in `/data/`:
   - `active_treasury_curve.csv`: Treasury yield curve
   - `sofr_curve.csv`: SOFR discount curve
   - `swaption_vols.csv`: Swaption volatilities for calibration

5. **Performance Targets**: 
   - 50,000 paths in < 10 seconds (initial target)
   - European option should match Black-Scholes within 0.1%
   - American value ≥ European value

## Documentation

Refer to `/docs/` for comprehensive documentation:
- `ls-technical-implementation.md`: Detailed implementation plan with code examples
- `ls-method-quant-doc.md`: Mathematical theory and background

The technical implementation doc provides phase-by-phase guidance and should be followed closely.