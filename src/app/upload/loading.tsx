import { Loader2 } from "lucide-react";

export default function UploadLoading() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20 flex items-center justify-center">
      <div className="text-center space-y-4">
        <Loader2 className="h-12 w-12 animate-spin mx-auto text-primary" />
        <div className="space-y-2">
          <h2 className="text-xl font-semibold">Loading Deepfake Detective</h2>
          <p className="text-muted-foreground">Preparing AI analysis tools...</p>
        </div>
      </div>
    </div>
  );
}
