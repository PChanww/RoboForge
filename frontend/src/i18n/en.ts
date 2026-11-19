/**
 * English dictionary, merged from one module per surface so each file stays
 * reviewable. Keys are the Chinese source strings used in components.
 * Later spreads win, so a shared word can be overridden per page if needed.
 */
import { BACKEND_LABELS } from './en/backend-labels'
import { EVALUATIONS } from './en/evaluations'
import { SHELL } from './en/shell'

export const EN: Record<string, string> = {
  ...BACKEND_LABELS,
  ...SHELL,
  ...EVALUATIONS,
}
