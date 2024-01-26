# frozen_string_literal: true

require 'bundler/gem_tasks'
require 'rake/testtask'

Rake::TestTask.new(:test) do |t|
  t.libs << 'test'
  t.libs << 'lib'
  t.test_files = FileList['test/**/test_*.rb']
end

require 'rake/extensiontask'

task build: :compile # rubocop:disable Rake/Desc

desc 'Run clang-format'
task :'clang-format' do
  sh 'clang-format -style=file -Werror --dry-run ext/numo/tiny_linalg/*.cpp ext/numo/tiny_linalg/*.hpp'
end

Rake::ExtensionTask.new('tiny_linalg') do |ext|
  ext.ext_dir = 'ext/numo/tiny_linalg'
  ext.lib_dir = 'lib/numo/tiny_linalg'
end

task default: %i[clobber compile test]
