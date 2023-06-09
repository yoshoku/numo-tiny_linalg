# Numo::TinyLinalg

[![Build Status](https://github.com/yoshoku/numo-tiny_linalg/actions/workflows/main.yml/badge.svg)](https://github.com/yoshoku/numo-tiny_linalg/actions/workflows/main.yml)
[![BSD 3-Clause License](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://github.com/yoshoku/numo-tiny_linalg/blob/main/LICENSE.txt)

Numo::TinyLinalg is a subset library from Numo::Linalg consisting only of methods used in Machine Learning algorithms.

This gem is still **under development** and may undergo many changes in the future.

## Installation

Install the OpenBlas.

macOS:

```sh
$ brew install openblas
```

Ubuntu:

```sh
$ sudo apt-get install libopenblas-dev liblapacke-dev
```

Install the gem and add to the application's Gemfile by executing.

macOS:

```sh
$ bundle config --local build.numo-tiny_linalg "--with-opt-dir=/opt/homebrew/Cellar/openblas/0.3.23/"
$ bundle add numo-tiny_linalg
```

Ubuntu:

```sh
$ bundle add numo-tiny_linalg
```

If bundler is not being used to manage dependencies, install the gem by executing.

macOS:

```sh
$ gem install numo-tiny_linalg -- --with-opt-dir=/opt/homebrew/Cellar/openblas/0.3.23/
```

Ubuntu:

```sh
$ gem install numo-tiny_linalg
```

## Usage

An example of singular value decomposition.

```ruby
require 'numo/tiny_linalg'

x = Numo::DFloat.new(5, 2).rand.dot(Numo::DFloat.new(2, 3).rand)
# =>
# Numo::DFloat#shape=[5,3]
# [[0.104945, 0.0284236, 0.117406],
#  [0.862634, 0.210945, 0.922135],
#  [0.324507, 0.0752655, 0.339158],
#  [0.67085, 0.102594, 0.600882],
#  [0.404631, 0.116868, 0.46644]]

s, u, vt = Numo::TinyLinalg.svd(x, job: 'S')

z = u.dot(s.diag).dot(vt)
# =>
# Numo::DFloat#shape=[5,3]
# [[0.104945, 0.0284236, 0.117406],
#  [0.862634, 0.210945, 0.922135],
#  [0.324507, 0.0752655, 0.339158],
#  [0.67085, 0.102594, 0.600882],
#  [0.404631, 0.116868, 0.46644]]

puts (x - z).abs.max
# => 4.440892098500626e-16
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/yoshoku/numo-tiny_linalg.
This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [code of conduct](https://github.com/yoshoku/numo-tiny_linalg/blob/main/CODE_OF_CONDUCT.md).

## Code of Conduct

Everyone interacting in the Numo::TinyLinalg project's codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/yoshoku/numo-tiny_linalg/blob/main/CODE_OF_CONDUCT.md).

## License

The gem is available as open source under the terms of the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
