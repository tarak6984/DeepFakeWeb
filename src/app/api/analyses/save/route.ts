import { NextResponse } from "next/server";
import { getServerSession } from "next-auth/next";
import { authOptions } from "@/lib/auth";
import { databaseService } from "@/lib/database-service";

export async function POST(req: Request) {
  try {
    // Check authentication
    const session = await getServerSession(authOptions);
    if (!session?.user?.id) {
      return NextResponse.json(
        { error: "Authentication required" },
        { status: 401 }
      );
    }

    // Parse request body
    const data = await req.json();
    
    // Validate required fields
    const requiredFields = [
      'filename', 'originalName', 'fileType', 'fileSize', 
      'confidence', 'prediction', 'processingTime', 'analysisId'
    ];
    
    for (const field of requiredFields) {
      if (data[field] === undefined || data[field] === null) {
        return NextResponse.json(
          { error: `Missing required field: ${field}` },
          { status: 400 }
        );
      }
    }

    // Save analysis to database
    const savedAnalysis = await databaseService.saveAnalysis({
      userId: session.user.id,
      filename: data.filename,
      originalName: data.originalName,
      fileType: data.fileType,
      fileSize: data.fileSize,
      confidence: data.confidence,
      prediction: data.prediction,
      fakeConfidence: data.fakeConfidence || 0,
      realConfidence: data.realConfidence || 0,
      processingTime: data.processingTime,
      modelsUsed: Array.isArray(data.modelsUsed) ? data.modelsUsed : ['EfficientNet', 'Xception'],
      detailedResults: data.detailedResults || {},
      explanation: data.explanation || {},
      analysisId: data.analysisId,
      duration: data.duration,
      resolution: data.resolution,
      frameCount: data.frameCount,
      sampleRate: data.sampleRate,
    });

    return NextResponse.json({
      success: true,
      analysis: {
        id: savedAnalysis.id,
        analysisId: savedAnalysis.analysisId,
        createdAt: savedAnalysis.createdAt,
      }
    });

  } catch (error) {
    console.error('Error saving analysis:', error);
    
    return NextResponse.json(
      { 
        error: "Failed to save analysis",
        details: error instanceof Error ? error.message : "Unknown error"
      },
      { status: 500 }
    );
  }
}