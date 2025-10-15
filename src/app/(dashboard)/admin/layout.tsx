import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Admin Panel | ITL Deepfake Detective',
  description: 'Administrative panel for managing users, system statistics, and platform settings',
};

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="admin-layout">
      {children}
    </div>
  );
}