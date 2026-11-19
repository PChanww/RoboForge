import type { ReactNode } from 'react'
import { useI18n } from '../i18n'

export default function ExperimentWorkspace({ title, description, actions, navigation, children }: {
  title: string
  description?: string
  actions?: ReactNode
  navigation?: ReactNode
  children: ReactNode
}) {
  const { t } = useI18n()
  return (
    <section className="workspace-page experiment-workspace">
      <header className="workspace-page-heading experiment-heading">
        <div>
          <h1>{t(title)}</h1>
          {description && <p>{t(description)}</p>}
        </div>
        {actions && <div className="experiment-actions">{actions}</div>}
      </header>
      {navigation}
      <div className="experiment-surface">{children}</div>
    </section>
  )
}
