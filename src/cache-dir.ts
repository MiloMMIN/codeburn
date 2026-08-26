import { homedir } from 'os'
import { join } from 'path'

/**
 * Resolve CodeBurn's shared cache directory at call time.
 *
 * Reading the environment on every call matters for embedded consumers and
 * tests that change CODEBURN_CACHE_DIR after importing the CLI modules.
 */
export function getCodeburnCacheDir(): string {
  const override = process.env['CODEBURN_CACHE_DIR']
  return override?.trim() ? override : join(homedir(), '.cache', 'codeburn')
}
