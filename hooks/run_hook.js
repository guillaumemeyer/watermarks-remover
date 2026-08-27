#!/usr/bin/env node
/**
 * Cross-platform launcher for watermarks-remover Claude Code plugin hook.
 * Claude Code runs on Node.js across Windows, macOS, and Linux.
 * This runner dynamically detects an available Python 3 interpreter
 * ('py -3', 'python3', or 'python') to avoid hardcoding issues on Windows.
 */

const { spawn } = require('child_process');
const path = require('path');

const script = path.resolve(__dirname, '..', 'service', 'scripts', 'hook_written_file.py');

const isWin = process.platform === 'win32';
const candidates = isWin
  ? [
      ['py', ['-3']],
      ['python', []],
      ['python3', []],
    ]
  : [
      ['python3', []],
      ['python', []],
    ];

function tryLaunch(index) {
  if (index >= candidates.length) {
    console.error(
      'watermarks-remover: could not find a working Python interpreter (tried py, python, python3)'
    );
    process.exit(1);
  }

  const [cmd, extraArgs] = candidates[index];
  const args = [...extraArgs, script, ...process.argv.slice(2)];

  const child = spawn(cmd, args, {
    stdio: 'inherit',
    windowsHide: true,
  });

  child.on('error', (err) => {
    if (err.code === 'ENOENT') {
      tryLaunch(index + 1);
    } else {
      console.error(`watermarks-remover: error running ${cmd}: ${err.message}`);
      process.exit(1);
    }
  });

  child.on('close', (code) => {
    process.exit(code ?? 0);
  });
}

tryLaunch(0);
