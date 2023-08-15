# frozen_string_literal: true

require_relative 'lib/numo/tiny_linalg/version'

Gem::Specification.new do |spec|
  spec.name = 'numo-tiny_linalg'
  spec.version = Numo::TinyLinalg::VERSION
  spec.authors = ['yoshoku']
  spec.email = ['yoshoku@outlook.com']

  spec.summary = <<~MSG
    Numo::TinyLinalg is a subset library from Numo::Linalg consisting only of methods used in Machine Learning algorithms.
  MSG
  spec.description = <<~MSG
    Numo::TinyLinalg is a subset library from Numo::Linalg consisting only of methods used in Machine Learning algorithms.
    The functions Numo::TinyLinalg supports are dot, det, eigh, inv, pinv, qr, solve, cholesky, cho_solve and svd.
  MSG
  spec.homepage = 'https://github.com/yoshoku/numo-tiny_linalg'
  spec.license = 'BSD-3-Clause'

  spec.metadata['homepage_uri'] = spec.homepage
  spec.metadata['source_code_uri'] = spec.homepage
  spec.metadata['changelog_uri'] = "#{spec.homepage}/blob/main/CHANGELOG.md"
  spec.metadata['documentation_uri'] = 'https://yoshoku.github.io/numo-tiny_linalg/doc/'

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  spec.files = Dir.chdir(__dir__) do
    `git ls-files -z`.split("\x0")
                     .reject { |f| f.match(%r{\A(?:(?:test|doc|node_modules|pkg|tmp|\.git|\.github|\.husky)/)}) }
                     .select { |f| f.match(/\.(?:rb|rbs|h|hpp|c|cpp|md|txt)$/) }
  end
  spec.files << 'vendor/tmp/.gitkeep'
  spec.bindir = 'exe'
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ['lib']
  spec.extensions = ['ext/numo/tiny_linalg/extconf.rb']

  # Uncomment to register a new dependency of your gem
  spec.add_dependency 'numo-narray', '>= 0.9.1'

  # For more information and examples about making a new gem, check out our
  # guide at: https://bundler.io/guides/creating_gem.html
  spec.metadata['rubygems_mfa_required'] = 'true'
end
