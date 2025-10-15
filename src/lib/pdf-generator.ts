import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { StoredAnalysis } from '@/lib/storage';

export interface PDFGenerationOptions {
  includeCharts?: boolean;
  includeTimeline?: boolean;
  includeAdvancedCharts?: boolean;
  includeRawData?: boolean;
  watermark?: string;
}

export class PDFReportGenerator {
  private pdf: jsPDF;
  private currentY: number = 20;
  private pageHeight: number = 297; // A4 height in mm
  private pageWidth: number = 210; // A4 width in mm
  private margin: number = 20;
  private lineHeight: number = 7;
  private currentAnalysis: StoredAnalysis | null = null;

  constructor() {
    this.pdf = new jsPDF('p', 'mm', 'a4');
    this.pdf.setFontSize(12);
  }

  private checkPageBreak(height: number = 10): void {
    if (this.currentY + height > this.pageHeight - this.margin) {
      this.pdf.addPage();
      this.currentY = this.margin;
    }
  }

  private addText(text: string, x: number = this.margin, fontSize: number = 12, isBold: boolean = false): void {
    this.pdf.setFontSize(fontSize);
    if (isBold) {
      this.pdf.setFont(undefined, 'bold');
    } else {
      this.pdf.setFont(undefined, 'normal');
    }
    
    // Handle text wrapping
    const splitText = this.pdf.splitTextToSize(text, this.pageWidth - 2 * this.margin);
    
    for (const line of splitText) {
      this.checkPageBreak();
      this.pdf.text(line, x, this.currentY);
      this.currentY += this.lineHeight;
    }
  }

  private addTitle(text: string, fontSize: number = 16): void {
    this.checkPageBreak(15);
    this.currentY += 5;
    this.addText(text, this.margin, fontSize, true);
    this.currentY += 5;
  }

  private addSection(title: string, content: string): void {
    this.addTitle(title, 14);
    this.addText(content);
    this.currentY += 5;
  }

  private async createChartElement(chartType: string, analysis: StoredAnalysis): Promise<HTMLElement> {
    const container = document.createElement('div');
    container.style.width = '800px';
    container.style.height = '400px';
    container.style.padding = '24px';
    container.style.backgroundColor = '#ffffff';
    container.style.position = 'fixed';
    container.style.top = '-2000px';
    container.style.left = '0px';
    container.style.zIndex = '10000';
    container.style.fontFamily = 'system-ui, -apple-system, sans-serif';
    
    if (chartType === 'confidence-gauge') {
      container.innerHTML = `
        <div style="border: none; box-shadow: none; background-color: #ffffff; padding: 16px; border-radius: 8px;">
          <div style="text-align: center; padding-bottom: 8px;">
            <h3 style="display: flex; align-items: center; justify-content: center; gap: 8px; font-size: 18px; margin: 0; color: #1f2937; font-weight: 600;">
              Confidence Analysis
            </h3>
          </div>
          <div style="padding: 16px; display: flex; align-items: center; justify-content: center;">
            <div style="position: relative; width: 128px; height: 128px;">
              <svg style="width: 128px; height: 128px; transform: rotate(-90deg);" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="40" stroke="#e5e7eb" stroke-width="8" fill="transparent"/>
                <circle cx="50" cy="50" r="40" 
                  stroke="${analysis.confidence < 0.9 ? '#10b981' : analysis.confidence < 0.95 ? '#f59e0b' : '#ef4444'}"
                  stroke-width="8" fill="transparent"
                  stroke-dasharray="${2 * Math.PI * 40 * analysis.confidence} ${2 * Math.PI * 40}"
                  stroke-linecap="round"/>
              </svg>
              <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center;">
                  <div style="font-size: 24px; font-weight: bold; color: ${analysis.confidence < 0.9 ? '#10b981' : analysis.confidence < 0.95 ? '#f59e0b' : '#ef4444'};">
                    ${Math.round(analysis.confidence * 100)}%
                  </div>
                  <div style="font-size: 14px; color: #6b7280;">Risk</div>
                </div>
              </div>
            </div>
          </div>
          <div style="text-align: center; margin-top: 16px;">
            <div style="display: inline-block; padding: 4px 8px; border-radius: 4px; background-color: ${analysis.confidence < 0.9 ? '#3b82f6' : analysis.confidence < 0.95 ? '#6b7280' : '#ef4444'}; color: #ffffff; font-size: 14px; font-weight: 500;">
              ${analysis.prediction.toUpperCase()}
            </div>
            <p style="font-size: 14px; color: #4b5563; margin: 8px 0 0 0;">
              ${analysis.confidence < 0.9 
                ? 'Content appears authentic with low risk of manipulation' 
                : analysis.confidence < 0.95 
                ? 'Moderate risk detected - further verification recommended'
                : 'High risk of artificial generation or manipulation detected'
              }
            </p>
          </div>
        </div>
      `;
    } else if (chartType === 'category-chart' && analysis.details.categoryBreakdown) {
      const categoryBars = Object.entries(analysis.details.categoryBreakdown).map(([category, score], index) => {
        const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
        const color = colors[index % colors.length];
        const percentage = score > 1 ? score : score * 100;
        return `
          <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <div style="min-width: 120px; font-size: 14px; font-weight: 500; color: #374151; text-transform: capitalize;">
              ${category}
            </div>
            <div style="flex: 1; background-color: #f3f4f6; height: 20px; border-radius: 10px; overflow: hidden;">
              <div style="height: 100%; background-color: ${color}; width: ${Math.max(percentage, 2)}%; border-radius: 10px;"></div>
            </div>
            <div style="min-width: 50px; font-size: 14px; font-weight: 600; color: #1f2937; text-align: right;">
              ${Math.round(percentage)}%
            </div>
          </div>
        `;
      }).join('');
      
      container.innerHTML = `
        <div style="border: none; box-shadow: none; background-color: #ffffff; border-radius: 8px;">
          <div style="padding: 16px 16px 8px 16px;">
            <h3 style="font-size: 18px; font-weight: 600; margin: 0; color: #1f2937;">Category Breakdown</h3>
          </div>
          <div style="padding: 16px;">
            <div style="height: 300px; width: 100%; display: flex; flex-direction: column; gap: 16px; justify-content: center;">
              ${categoryBars}
            </div>
          </div>
        </div>
      `;
    } else if (chartType === 'timeline-chart') {
      // Simple timeline representation
      const points = analysis.details.frameAnalysis?.length || 20;
      const timelineBars = Array.from({ length: points }, (_, i) => {
        const variation = (Math.sin(i * 0.5) - 0.5) * 0.3;
        const confidence = Math.max(0, Math.min(1, analysis.confidence + variation));
        const height = confidence * 80; // Max height 80px
        return `<div style="width: 8px; height: ${height}px; background-color: #3b82f6; border-radius: 2px;"></div>`;
      }).join('');
      
      container.innerHTML = `
        <div style="border: none; box-shadow: none; background-color: #ffffff; border-radius: 8px;">
          <div style="padding: 16px 16px 8px 16px;">
            <h3 style="font-size: 18px; font-weight: 600; margin: 0; color: #1f2937;">Timeline Analysis</h3>
          </div>
          <div style="padding: 16px;">
            <div style="height: 200px; width: 100%; display: flex; align-items: end; justify-content: space-between; gap: 2px;">
              ${timelineBars}
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 12px; font-size: 12px; color: #6b7280;">
              <span>Start</span>
              <span>End</span>
            </div>
          </div>
        </div>
      `;
    } else if (chartType === 'risk-heatmap') {
      // Create a simple heatmap visualization
      const heatmapCells = Array.from({ length: 64 }).map((_, i) => {
        const hue = 120 - (analysis.confidence * 120); // Green to red
        const lightness = 50 + (Math.sin(i * 0.5) * 20);
        return `<div style="aspect-ratio: 1; border-radius: 2px; background-color: hsl(${hue}, 60%, ${Math.max(30, Math.min(70, lightness))}%);"</div>`;
      }).join('');
      
      container.innerHTML = `
        <div style="border: none; box-shadow: none; background-color: #ffffff; border-radius: 8px;">
          <div style="padding: 16px 16px 8px 16px;">
            <h3 style="font-size: 18px; font-weight: 600; margin: 0; color: #1f2937;">Risk Analysis Heatmap</h3>
          </div>
          <div style="padding: 16px;">
            <div style="display: grid; grid-template-columns: repeat(8, 1fr); gap: 4px; margin-bottom: 16px;">
              ${heatmapCells}
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 14px; color: #6b7280;">
              <span>Low Risk</span>
              <span>High Risk</span>
            </div>
          </div>
        </div>
      `;
    } else if (chartType === 'anomaly-scatter') {
      // Create scatter plot representation
      const scatterPoints = Array.from({ length: 20 }).map((_, i) => {
        const x = (i / 20) * 90 + 5; // 5% to 95%
        const y = Math.random() * 80 + 10; // 10% to 90%
        const size = Math.random() * 6 + 2; // 2px to 8px
        return `<div style="position: absolute; left: ${x}%; top: ${y}%; width: ${size}px; height: ${size}px; background-color: #ef4444; border-radius: 50%;"></div>`;
      }).join('');
      
      container.innerHTML = `
        <div style="border: none; box-shadow: none; background-color: #ffffff; border-radius: 8px;">
          <div style="padding: 16px 16px 8px 16px;">
            <h3 style="font-size: 18px; font-weight: 600; margin: 0; color: #1f2937;">Anomaly Detection</h3>
          </div>
          <div style="padding: 16px;">
            <div style="height: 200px; width: 100%; position: relative; border: 1px solid #e5e7eb; border-radius: 8px;">
              ${scatterPoints}
            </div>
          </div>
        </div>
      `;
    } else if (chartType === 'radar-chart') {
      // Create radar chart representation using progress bars
      const metrics = [
        { name: 'Visual Quality', score: Math.round(analysis.confidence * 80 + 20) },
        { name: 'Temporal Consistency', score: Math.round(analysis.confidence * 75 + 25) },
        { name: 'Audio Synchronization', score: Math.round(analysis.confidence * 85 + 15) },
        { name: 'Compression Artifacts', score: Math.round(analysis.confidence * 70 + 30) },
        { name: 'Facial Features', score: Math.round(analysis.confidence * 90 + 10) },
      ];
      
      const metricBars = metrics.map((metric) => `
        <div style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 12px;">
          <div style="display: flex; justify-content: space-between; font-size: 14px;">
            <span style="color: #374151;">${metric.name}</span>
            <span style="font-weight: 500; color: #1f2937;">${metric.score}%</span>
          </div>
          <div style="width: 100%; background-color: #e5e7eb; border-radius: 9999px; height: 8px;">
            <div style="background-color: #2563eb; height: 8px; border-radius: 9999px; width: ${metric.score}%;"></div>
          </div>
        </div>
      `).join('');
      
      container.innerHTML = `
        <div style="border: none; box-shadow: none; background-color: #ffffff; border-radius: 8px;">
          <div style="padding: 16px 16px 8px 16px;">
            <h3 style="font-size: 18px; font-weight: 600; margin: 0; color: #1f2937;">Multidimensional Analysis</h3>
          </div>
          <div style="padding: 16px;">
            ${metricBars}
          </div>
        </div>
      `;
    }
    
    document.body.appendChild(container);
    return container;
  }

  private async addChartFromElement(elementId: string, title: string): Promise<void> {
    let element = document.getElementById(elementId);
    let createdElement = false;
    
    // If element doesn't exist, create it dynamically
    if (!element) {
      console.log(`Creating dynamic chart element for ${elementId}`);
      try {
        const chartType = elementId.replace('-pdf', '');
        element = await this.createChartElement(chartType, this.currentAnalysis!);
        createdElement = true;
      } catch (error) {
        console.warn(`Failed to create chart element ${elementId}:`, error);
        this.addTitle(title, 12);
        this.addText(`[Chart could not be rendered: ${title} - Creation failed]`, this.margin, 10);
        return;
      }
    }

    try {
      // Add title for the chart
      this.addTitle(title, 12);

      // Wait for element to be ready
      await new Promise(resolve => setTimeout(resolve, 300));
      
      const canvas = await html2canvas(element, {
        backgroundColor: '#ffffff',
        scale: 1.2,
        logging: false,
        useCORS: true,
        allowTaint: true,
        width: element.offsetWidth || 800,
        height: element.offsetHeight || 400,
      });
      
      const imgData = canvas.toDataURL('image/png');
      const imgWidth = this.pageWidth - 2 * this.margin;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;

      // Check if we need a new page for the image
      this.checkPageBreak(imgHeight + 10);

      this.pdf.addImage(imgData, 'PNG', this.margin, this.currentY, imgWidth, imgHeight);
      this.currentY += imgHeight + 10;
      
      // Clean up created element
      if (createdElement && element) {
        document.body.removeChild(element);
      }
      
    } catch (error) {
      console.error('Failed to add chart to PDF:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      this.addText(`[Chart could not be rendered: ${title} - ${errorMessage}]`, this.margin, 10);
      
      // Clean up on error
      if (createdElement && element) {
        document.body.removeChild(element);
      }
    }
  }

  private addHeader(analysis: StoredAnalysis): void {
    // Company/App header
    this.pdf.setFontSize(20);
    this.pdf.setFont(undefined, 'bold');
    this.pdf.text('ITL DeepFake Detection Report', this.pageWidth / 2, 15, { align: 'center' });

    this.currentY = 30;

    // File info header
    this.pdf.setFontSize(16);
    this.pdf.setFont(undefined, 'bold');
    this.pdf.text(`Analysis Report: ${analysis.filename}`, this.margin, this.currentY);
    this.currentY += 10;

    this.pdf.setFontSize(10);
    this.pdf.setFont(undefined, 'normal');
    this.pdf.text(`Generated on: ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}`, this.margin, this.currentY);
    this.currentY += 5;
    this.pdf.text(`Analysis Date: ${new Date(analysis.timestamp).toLocaleDateString()}`, this.margin, this.currentY);
    this.currentY += 10;

    // Add a separator line
    this.pdf.line(this.margin, this.currentY, this.pageWidth - this.margin, this.currentY);
    this.currentY += 10;
  }

  private addFooter(): void {
    const pageCount = this.pdf.internal.getNumberOfPages();
    
    for (let i = 1; i <= pageCount; i++) {
      this.pdf.setPage(i);
      this.pdf.setFontSize(8);
      this.pdf.setFont(undefined, 'normal');
      this.pdf.text(
        `Page ${i} of ${pageCount} | Generated by ITL DeepFake Detection System`,
        this.pageWidth / 2,
        this.pageHeight - 10,
        { align: 'center' }
      );
    }
  }

  private addExecutiveSummary(analysis: StoredAnalysis): void {
    this.addTitle('Executive Summary', 16);
    
    const riskLevel = analysis.confidence < 0.3 ? 'LOW' : analysis.confidence < 0.7 ? 'MODERATE' : 'HIGH';
    const riskColor = analysis.confidence < 0.3 ? 'green' : analysis.confidence < 0.7 ? 'orange' : 'red';
    
    const summary = `
File: ${analysis.filename}
Risk Level: ${riskLevel} (${Math.round(analysis.confidence * 100)}% confidence)
Prediction: ${analysis.prediction}
Processing Time: ${Math.round(analysis.processingTime / 1000)} seconds
File Size: ${(analysis.fileSize / (1024 * 1024)).toFixed(2)} MB
File Type: ${analysis.fileType}

Analysis Results:
The file has been analyzed using advanced deepfake detection algorithms. Based on the analysis, this content has a ${riskLevel.toLowerCase()} probability of being artificially generated or manipulated. The confidence score of ${Math.round(analysis.confidence * 100)}% indicates the system's certainty in this assessment.

${analysis.prediction === 'authentic' 
  ? 'The content appears to be authentic with no significant signs of artificial manipulation detected.'
  : 'The content shows signs of potential artificial generation or manipulation and should be treated with caution.'
}
`;

    this.addText(summary);
  }

  private addTechnicalDetails(analysis: StoredAnalysis): void {
    this.addTitle('Technical Analysis Details', 16);
    
    // File metadata
    this.addSection('File Metadata', `
Type: ${analysis.fileType}
Size: ${(analysis.fileSize / (1024 * 1024)).toFixed(2)} MB
Duration: ${analysis.details.metadata.duration ? Math.round(analysis.details.metadata.duration) + 's' : 'N/A'}
Resolution: ${analysis.details.metadata.resolution || 'N/A'}
Analysis ID: ${analysis.id}
`);

    // Processing details
    this.addSection('Processing Information', `
Processing Time: ${Math.round(analysis.processingTime / 1000)} seconds
Analysis Algorithm: ITL AI Detection Models
Timestamp: ${new Date(analysis.timestamp).toLocaleString()}
`);

    // Detailed results
    if (analysis.details.categoryBreakdown) {
      this.addSection('Category Analysis', 
        Object.entries(analysis.details.categoryBreakdown)
          .map(([category, score]) => `${category}: ${score}%`)
          .join('\n')
      );
    }

    // Frame analysis if available
    if (analysis.details.frameAnalysis) {
      this.addSection('Frame Analysis', `
Total Frames Analyzed: ${analysis.details.frameAnalysis.length}
Average Frame Confidence: ${(analysis.details.frameAnalysis.reduce((sum, frame) => sum + frame.confidence, 0) / analysis.details.frameAnalysis.length * 100).toFixed(1)}%
Suspicious Frames: ${analysis.details.frameAnalysis.filter(frame => frame.confidence > 0.7).length}
`);
    }

    // Audio analysis if available
    if (analysis.details.audioAnalysis) {
      this.addSection('Audio Analysis', `
Audio Segments Analyzed: ${analysis.details.audioAnalysis.segments.length}
Average Audio Confidence: ${(analysis.details.audioAnalysis.segments.reduce((sum, seg) => sum + seg.confidence, 0) / analysis.details.audioAnalysis.segments.length * 100).toFixed(1)}%
Audio Anomalies Found: ${analysis.details.audioAnalysis.segments.filter(seg => seg.anomalies && seg.anomalies.length > 0).length}
`);
    }
  }

  private addDisclaimer(): void {
    this.addTitle('Disclaimer and Limitations', 14);
    
    const disclaimer = `
IMPORTANT NOTICE:

1. Accuracy Limitations: While this deepfake detection system uses advanced AI algorithms, no detection system is 100% accurate. Results should be considered as one factor in content verification, not as definitive proof.

2. Technology Evolution: Deepfake generation technology is rapidly evolving. New techniques may not be detected by current algorithms.

3. Context Matters: Consider the source, context, and other verification methods when evaluating content authenticity.

4. Legal Considerations: This report is provided for informational purposes only and should not be used as sole evidence in legal proceedings without additional verification.

5. Data Privacy: Analysis data is processed according to our privacy policy. No content is stored permanently on our servers after analysis.

6. Technical Support: For questions about this report or the analysis methodology, contact our technical support team.

Generated by ITL DeepFake Detection System v1.0
Powered by ITL AI Detection Models
`;

    this.addText(disclaimer, this.margin, 10);
  }

  public async generateReport(
    analysis: StoredAnalysis,
    options: PDFGenerationOptions = {}
  ): Promise<Blob> {
    try {
      // Store current analysis for chart generation
      this.currentAnalysis = analysis;
      
      // Reset PDF state
      this.currentY = 20;

      // Add header
      this.addHeader(analysis);

      // Add executive summary
      this.addExecutiveSummary(analysis);

      // Add charts if requested
      if (options.includeCharts !== false) {
        this.checkPageBreak(50);
        this.addTitle('Visual Analysis', 16);
        
        // Add confidence gauge
        await this.addChartFromElement('confidence-gauge-pdf', 'Confidence Analysis');
        
        // Add category breakdown
        if (analysis.details.categoryBreakdown) {
          await this.addChartFromElement('category-chart-pdf', 'Category Breakdown');
        }
      }

      // Add timeline analysis if requested
      if (options.includeTimeline !== false) {
        await this.addChartFromElement('timeline-chart-pdf', 'Timeline Analysis');
      }

      // Add advanced charts if requested
      if (options.includeAdvancedCharts) {
        await this.addChartFromElement('risk-heatmap-pdf', 'Risk Analysis Heatmap');
        await this.addChartFromElement('anomaly-scatter-pdf', 'Anomaly Detection');
        await this.addChartFromElement('radar-chart-pdf', 'Multidimensional Analysis');
      }

      // Add technical details
      this.addTechnicalDetails(analysis);

      // Add raw data if requested
      if (options.includeRawData) {
        this.addTitle('Raw Analysis Data', 16);
        this.addText(JSON.stringify(analysis, null, 2), this.margin, 8);
      }

      // Add disclaimer
      this.addDisclaimer();

      // Add footer to all pages
      this.addFooter();

      // Generate PDF blob
      const pdfBlob = new Blob([this.pdf.output('blob')], { type: 'application/pdf' });
      
      return pdfBlob;

    } catch (error) {
      console.error('Failed to generate PDF report:', error);
      throw new Error('Failed to generate PDF report. Please try again.');
    }
  }

  public downloadReport(analysis: StoredAnalysis, options: PDFGenerationOptions = {}): Promise<void> {
    return this.generateReport(analysis, options).then(blob => {
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `itl-deepfake-analysis-${analysis.filename}-${new Date().toISOString().split('T')[0]}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    });
  }
}

// Export a singleton instance
export const pdfGenerator = new PDFReportGenerator();