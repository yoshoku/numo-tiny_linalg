# frozen_string_literal: true

if defined?(Numo::Linalg)
  warn 'Numo::Linalg is already defined. Numo::TinyLinalg will not be loaded.'
else
  require 'numo/narray'
  require_relative 'tiny_linalg/version'
  require_relative 'tiny_linalg/tiny_linalg'

  module Numo
    module TinyLinalg
      class Error < StandardError; end
      # Your code goes here...
    end

    # Linalg = TinyLinalg
  end
end
