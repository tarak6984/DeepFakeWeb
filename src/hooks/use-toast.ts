import { toast as sonnerToast } from "sonner"

interface ToastOptions {
  title?: string
  description?: string
  variant?: 'default' | 'destructive' | 'success'
  duration?: number
}

function toast({ title, description, variant = 'default', duration }: ToastOptions) {
  const message = title || description || 'Notification'
  const desc = title && description ? description : undefined

  switch (variant) {
    case 'destructive':
      return sonnerToast.error(message, { description: desc, duration })
    case 'success':
      return sonnerToast.success(message, { description: desc, duration })
    default:
      return sonnerToast(message, { description: desc, duration })
  }
}

function useToast() {
  return {
    toast,
    dismiss: sonnerToast.dismiss,
  }
}

export { useToast, toast }
