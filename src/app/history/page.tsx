'use client'

import { useState, useEffect } from 'react'
import { useSession } from 'next-auth/react'
import { useRouter } from 'next/navigation'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { 
  Search,
  Filter,
  ArrowLeft,
  Calendar,
  FileText,
  Image as ImageIcon,
  Video,
  Volume2,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Download,
  Eye,
  MoreHorizontal,
  Clock,
  Archive
} from 'lucide-react'
import { motion } from 'framer-motion'
import { format } from 'date-fns'
import Link from 'next/link'

interface Analysis {
  id: string
  filename: string
  fileType: string
  fileSize: number
  prediction: 'AUTHENTIC' | 'MANIPULATED' | 'INCONCLUSIVE'
  confidence: number
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH'
  mediaType: 'IMAGE' | 'AUDIO' | 'VIDEO'
  processingTime: number
  createdAt: string
  modelsUsed: string[]
}

export default function HistoryPage() {
  const { data: session, status } = useSession()
  const router = useRouter()
  const [analyses, setAnalyses] = useState<Analysis[]>([])
  const [filteredAnalyses, setFilteredAnalyses] = useState<Analysis[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedFilter, setSelectedFilter] = useState<string>('all')
  const [sortBy, setSortBy] = useState<'newest' | 'oldest' | 'confidence'>('newest')

  useEffect(() => {
    if (status === 'loading') return
    if (status === 'unauthenticated') {
      router.push('/auth/signin')
      return
    }

    fetchAnalyses()
  }, [status, router])

  useEffect(() => {
    filterAndSortAnalyses()
  }, [analyses, searchQuery, selectedFilter, sortBy])

  const fetchAnalyses = async () => {
    try {
      const response = await fetch('/api/user/analyses')
      if (response.ok) {
        const data = await response.json()
        setAnalyses(data.analyses || [])
      }
    } catch (error) {
      console.error('Error fetching analyses:', error)
    } finally {
      setLoading(false)
    }
  }

  const filterAndSortAnalyses = () => {
    let filtered = [...analyses]

    // Search filter
    if (searchQuery) {
      filtered = filtered.filter(analysis =>
        analysis.filename.toLowerCase().includes(searchQuery.toLowerCase())
      )
    }

    // Type filter
    if (selectedFilter !== 'all') {
      filtered = filtered.filter(analysis => {
        switch (selectedFilter) {
          case 'images':
            return analysis.mediaType === 'IMAGE'
          case 'audio':
            return analysis.mediaType === 'AUDIO'
          case 'video':
            return analysis.mediaType === 'VIDEO'
          case 'authentic':
            return analysis.prediction === 'AUTHENTIC'
          case 'manipulated':
            return analysis.prediction === 'MANIPULATED'
          case 'inconclusive':
            return analysis.prediction === 'INCONCLUSIVE'
          case 'high-risk':
            return analysis.riskLevel === 'HIGH'
          default:
            return true
        }
      })
    }

    // Sort
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'newest':
          return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
        case 'oldest':
          return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
        case 'confidence':
          return b.confidence - a.confidence
        default:
          return 0
      }
    })

    setFilteredAnalyses(filtered)
  }

  const getMediaIcon = (mediaType: string) => {
    switch (mediaType) {
      case 'IMAGE':
        return <ImageIcon className="w-4 h-4" />
      case 'AUDIO':
        return <Volume2 className="w-4 h-4" />
      case 'VIDEO':
        return <Video className="w-4 h-4" />
      default:
        return <FileText className="w-4 h-4" />
    }
  }

  const getPredictionIcon = (prediction: string) => {
    switch (prediction) {
      case 'AUTHENTIC':
        return <CheckCircle className="w-4 h-4 text-green-600" />
      case 'MANIPULATED':
        return <XCircle className="w-4 h-4 text-red-600" />
      case 'INCONCLUSIVE':
        return <AlertTriangle className="w-4 h-4 text-yellow-600" />
      default:
        return <AlertTriangle className="w-4 h-4 text-gray-600" />
    }
  }

  const getPredictionColor = (prediction: string) => {
    switch (prediction) {
      case 'AUTHENTIC':
        return 'text-green-600 bg-green-50 border-green-200'
      case 'MANIPULATED':
        return 'text-red-600 bg-red-50 border-red-200'
      case 'INCONCLUSIVE':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const formatFileSize = (bytes: number) => {
    const sizes = ['B', 'KB', 'MB', 'GB']
    if (bytes === 0) return '0 B'
    const i = Math.floor(Math.log(bytes) / Math.log(1024))
    return `${Math.round(bytes / Math.pow(1024, i) * 100) / 100} ${sizes[i]}`
  }

  if (status === 'loading' || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading analysis history...</p>
        </div>
      </div>
    )
  }

  if (!session) return null

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/10">
      {/* Header */}
      <div className="border-b bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button variant="outline" size="sm" asChild>
                <Link href="/dashboard">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to Dashboard
                </Link>
              </Button>
              <div>
                <h1 className="text-2xl font-bold">Analysis History</h1>
                <p className="text-muted-foreground">
                  {analyses.length} total analyses • {filteredAnalyses.length} shown
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <Button variant="outline" size="sm">
                <Archive className="w-4 h-4 mr-2" />
                Export All
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="space-y-6">
          {/* Search and Filters */}
          <Card>
            <CardContent className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {/* Search */}
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
                  <Input
                    placeholder="Search by filename..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10"
                  />
                </div>

                {/* Filter by Type */}
                <select
                  className="px-3 py-2 border border-input bg-background rounded-md"
                  value={selectedFilter}
                  onChange={(e) => setSelectedFilter(e.target.value)}
                >
                  <option value="all">All Files</option>
                  <option value="images">Images Only</option>
                  <option value="audio">Audio Only</option>
                  <option value="video">Video Only</option>
                  <option value="authentic">Authentic</option>
                  <option value="manipulated">Manipulated</option>
                  <option value="inconclusive">Inconclusive</option>
                  <option value="high-risk">High Risk</option>
                </select>

                {/* Sort */}
                <select
                  className="px-3 py-2 border border-input bg-background rounded-md"
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                >
                  <option value="newest">Newest First</option>
                  <option value="oldest">Oldest First</option>
                  <option value="confidence">By Confidence</option>
                </select>

                <Button variant="outline">
                  <Filter className="w-4 h-4 mr-2" />
                  Advanced
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Analysis List */}
          {filteredAnalyses.length > 0 ? (
            <div className="space-y-4">
              {filteredAnalyses.map((analysis, index) => (
                <motion.div
                  key={analysis.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <Card className="hover:shadow-lg transition-shadow">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4 flex-1">
                          {/* File Icon */}
                          <div className="w-12 h-12 bg-muted/30 rounded-lg flex items-center justify-center">
                            {getMediaIcon(analysis.mediaType)}
                          </div>

                          {/* File Info */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <h3 className="font-medium text-sm truncate max-w-xs">
                                {analysis.filename}
                              </h3>
                              <Badge variant="outline" className="text-xs">
                                {analysis.mediaType.toLowerCase()}
                              </Badge>
                            </div>
                            <div className="flex items-center gap-4 text-xs text-muted-foreground">
                              <span>{formatFileSize(analysis.fileSize)}</span>
                              <span className="flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                {analysis.processingTime}ms
                              </span>
                              <span>{format(new Date(analysis.createdAt), 'MMM d, yyyy')}</span>
                            </div>
                          </div>

                          {/* Prediction */}
                          <div className="text-center">
                            <div className="flex items-center justify-center gap-2 mb-1">
                              {getPredictionIcon(analysis.prediction)}
                              <span className="text-sm font-medium">
                                {Math.round(analysis.confidence * 100)}%
                              </span>
                            </div>
                            <Badge 
                              variant="outline"
                              className={`text-xs ${getPredictionColor(analysis.prediction)}`}
                            >
                              {analysis.prediction.toLowerCase()}
                            </Badge>
                          </div>

                          {/* Risk Level */}
                          <div className="text-center">
                            <Badge 
                              variant={
                                analysis.riskLevel === 'HIGH' ? 'destructive' :
                                analysis.riskLevel === 'MEDIUM' ? 'secondary' : 'outline'
                              }
                              className="text-xs"
                            >
                              {analysis.riskLevel.toLowerCase()} risk
                            </Badge>
                          </div>

                          {/* Actions */}
                          <div className="flex items-center gap-2">
                            <Button variant="outline" size="sm" asChild>
                              <Link href={`/results?id=${analysis.id}`}>
                                <Eye className="w-4 h-4" />
                              </Link>
                            </Button>
                            <Button variant="outline" size="sm">
                              <Download className="w-4 h-4" />
                            </Button>
                          </div>
                        </div>
                      </div>

                      {/* Models Used */}
                      {Array.isArray(analysis.modelsUsed) && analysis.modelsUsed.length > 0 && (
                        <div className="mt-4 pt-4 border-t">
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <span>Models:</span>
                            {analysis.modelsUsed.map((model, i) => (
                              <Badge key={i} variant="outline" className="text-xs">
                                {model}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="py-12 text-center">
                <Archive className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">No analyses found</h3>
                <p className="text-muted-foreground mb-4">
                  {searchQuery || selectedFilter !== 'all' 
                    ? 'Try adjusting your search or filter criteria.'
                    : 'Start by uploading and analyzing your first file.'
                  }
                </p>
                {!searchQuery && selectedFilter === 'all' && (
                  <Button asChild>
                    <Link href="/upload">
                      Start Analyzing
                    </Link>
                  </Button>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}