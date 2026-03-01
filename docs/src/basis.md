# Basis / Dictionary Support

When a signal is not sparse in the canonical basis but is sparse under some dictionary or basis ``\Psi``, CompSense supports recovery via the transformation ``x = \Psi \theta`` where ``\theta`` is sparse.

Use [`recover_in_basis`](@ref) to transparently handle this transformation with any recovery algorithm.

## API Reference

```@docs
recover_in_basis
dct_matrix
identity_matrix
```
