/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkCorrelationImageToImageMetricv4HelperThreader_h
#define itkCorrelationImageToImageMetricv4HelperThreader_h

#include "itkImageToImageMetricv4GetValueAndDerivativeThreader.h"

#include <memory> // For unique_ptr.

namespace itk
{

/** \class CorrelationImageToImageMetricv4GetValueAndDerivativeThreader
 * \brief Helper class for CorrelationImageToImageMetricv4 \c
 * To compute the average pixel intensities of the fixed image and the moving image
 * on the sampled points or inside the virtual image region:
 * \f$ \bar f (CorrelationImageToImageMetricv4::m\_AverageFix ) \f$
 * \f$ \bar m (CorrelationImageToImageMetricv4::m\_AverageMov ) \f$
 *
 * \ingroup ITKMetricsv4
 */
template <typename TDomainPartitioner, typename TImageToImageMetric, typename TCorrelationMetric>
class ITK_TEMPLATE_EXPORT CorrelationImageToImageMetricv4HelperThreader
  : public ImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(CorrelationImageToImageMetricv4HelperThreader);

  /** Standard class type aliases. */
  using Self = CorrelationImageToImageMetricv4HelperThreader;
  using Superclass = ImageToImageMetricv4GetValueAndDerivativeThreader<TDomainPartitioner, TImageToImageMetric>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkOverrideGetNameOfClassMacro(CorrelationImageToImageMetricv4HelperThreader);

  itkNewMacro(Self);

  using typename Superclass::DomainType;
  using typename Superclass::AssociateType;

  using ImageToImageMetricv4Type = typename Superclass::ImageToImageMetricv4Type;
  using typename Superclass::VirtualIndexType;
  using typename Superclass::VirtualPointType;
  using typename Superclass::FixedImagePointType;
  using typename Superclass::FixedImagePixelType;
  using typename Superclass::FixedImageGradientType;
  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImageGradientType;
  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using typename Superclass::DerivativeValueType;

  using typename Superclass::InternalComputationValueType;
  using typename Superclass::NumberOfParametersType;

  using typename Superclass::FixedOutputPointType;
  using typename Superclass::MovingOutputPointType;

protected:
  CorrelationImageToImageMetricv4HelperThreader();
  ~CorrelationImageToImageMetricv4HelperThreader() override = default;

  /** Overload: Resize and initialize per thread objects. */
  void
  BeforeThreadedExecution() override;

  /** Overload:
   * Collects the results from each thread and sums them.  Results are stored
   * in the enclosing class \c m_Value and \c m_DerivativeResult.  Behavior
   * depends on m_AverageValueAndDerivativeByNumberOfValuePoints,
   * m_NumberOfValidPoints, to average the value sum, and to average
   * derivative sums for global transforms only (i.e. transforms without local
   * support).  */
  void
  AfterThreadedExecution() override;


  /* Overload: don't need to compute the image gradients and store derivatives
   *
   * Method called by the threaders to process the given virtual point.  This
   * in turn calls \c TransformAndEvaluateFixedPoint, \c
   * TransformAndEvaluateMovingPoint, and \c ProcessPoint.
   */
  bool
  ProcessVirtualPoint(const VirtualIndexType & virtualIndex,
                      const VirtualPointType & virtualPoint,
                      const ThreadIdType       threadId) override;


  /**
   * Not using. All processing is done in ProcessVirtualPoint.
   */
  bool
  ProcessPoint(const VirtualIndexType &,
               const VirtualPointType &,
               const FixedImagePointType &,
               const FixedImagePixelType &,
               const FixedImageGradientType &,
               const MovingImagePointType &,
               const MovingImagePixelType &,
               const MovingImageGradientType &,
               MeasureType &,
               DerivativeType &,
               const ThreadIdType) const override
  {
    return false;
  }

private:
  struct CorrelationMetricPerThreadStruct
  {
    InternalComputationValueType FixSum;
    InternalComputationValueType MovSum;
  };
  itkPadStruct(ITK_CACHE_LINE_ALIGNMENT, CorrelationMetricPerThreadStruct, PaddedCorrelationMetricPerThreadStruct);
  itkAlignedTypedef(ITK_CACHE_LINE_ALIGNMENT,
                    PaddedCorrelationMetricPerThreadStruct,
                    AlignedCorrelationMetricPerThreadStruct);
  /* per thread variables for correlation and its derivatives */
  std::unique_ptr<AlignedCorrelationMetricPerThreadStruct[]> m_CorrelationMetricPerThreadVariables;

  /** Internal pointer to the metric object in use by this threader.
   *  This will avoid costly dynamic casting in tight loops. */
  TCorrelationMetric * m_CorrelationAssociate{};
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCorrelationImageToImageMetricv4HelperThreader.hxx"
#endif

#endif
