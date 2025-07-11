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
#ifndef itkInverseDeconvolutionImageFilter_h
#define itkInverseDeconvolutionImageFilter_h

#include "itkFFTConvolutionImageFilter.h"
#include "itkMath.h"

namespace itk
{
/**
 * \class InverseDeconvolutionImageFilter
 * \brief The direct linear inverse deconvolution filter.
 *
 * The inverse filter is the most straightforward deconvolution
 * method. Considering that convolution of two images in the spatial domain is
 * equivalent to multiplying the Fourier transform of the two images,
 * the inverse filter consists of inverting the multiplication. In
 * other words, this filter computes the following:
 * \f[ \hat{F}(\omega) =
 *       \begin{cases}
 *         G(\omega) / H(\omega) & \text{if $|H(\omega)| \geq \epsilon$} \\
 *         0                     & \text{otherwise}
 *       \end{cases}
 * \f]
 * where \f$\hat{F}(\omega)\f$ is the Fourier transform of the
 * estimate produced by this filter, \f$G(\omega)\f$ is the Fourier
 * transform of the input blurred image, \f$H(\omega)\f$ is the
 * Fourier transform of the blurring kernel, and \f$\epsilon\f$ is a
 * constant real non-negative threshold (called
 * KernelZeroMagnitudeThreshold in this filter) that determines when
 * the magnitude of a complex number is considered zero.
 *
 * \author Gaetan Lehmann, Biologie du Developpement et de la Reproduction, INRA de Jouy-en-Josas, France
 * \author Cory Quammen, The University of North Carolina at Chapel Hill
 *
 * \ingroup ITKDeconvolution
 *
 */
template <typename TInputImage,
          typename TKernelImage = TInputImage,
          typename TOutputImage = TInputImage,
          typename TInternalPrecision = double>
class ITK_TEMPLATE_EXPORT InverseDeconvolutionImageFilter
  : public FFTConvolutionImageFilter<TInputImage, TKernelImage, TOutputImage, TInternalPrecision>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(InverseDeconvolutionImageFilter);

  using Self = InverseDeconvolutionImageFilter;
  using Superclass = FFTConvolutionImageFilter<TInputImage, TKernelImage, TOutputImage, TInternalPrecision>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** \see LightObject::GetNameOfClass() */
  itkOverrideGetNameOfClassMacro(InverseDeconvolutionImageFilter);

  /** Dimensionality of input and output data is assumed to be the same. */
  static constexpr unsigned int ImageDimension = TInputImage::ImageDimension;

  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using KernelImageType = TKernelImage;
  using typename Superclass::InputPixelType;
  using typename Superclass::OutputPixelType;
  using typename Superclass::KernelPixelType;
  using typename Superclass::InputIndexType;
  using typename Superclass::OutputIndexType;
  using typename Superclass::KernelIndexType;
  using typename Superclass::InputSizeType;
  using typename Superclass::OutputSizeType;
  using typename Superclass::KernelSizeType;
  using typename Superclass::SizeValueType;
  using typename Superclass::InputRegionType;
  using typename Superclass::OutputRegionType;
  using typename Superclass::KernelRegionType;

  /** Internal image types. */
  using typename Superclass::InternalImageType;
  using typename Superclass::InternalImagePointerType;
  using typename Superclass::InternalComplexType;
  using typename Superclass::InternalComplexImageType;
  using typename Superclass::InternalComplexImagePointerType;

  /** Set/get the threshold value used to determine whether a
   * frequency of the Fourier transform of the blurring kernel is
   * considered to be zero. Default value is 1.0e-4. */
  /** @ITKStartGrouping */
  itkSetMacro(KernelZeroMagnitudeThreshold, double);
  itkGetConstMacro(KernelZeroMagnitudeThreshold, double);
  /** @ITKEndGrouping */
protected:
  InverseDeconvolutionImageFilter();
  ~InverseDeconvolutionImageFilter() override = default;

  /** This filter uses a minipipeline to compute the output. */
  void
  GenerateData() override;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  double m_KernelZeroMagnitudeThreshold{};
};

namespace Functor
{
template <typename TInput1, typename TInput2, typename TOutput>
class ITK_TEMPLATE_EXPORT InverseDeconvolutionFunctor
{
public:
  InverseDeconvolutionFunctor() = default;
  ~InverseDeconvolutionFunctor() = default;

  bool
  operator==(const InverseDeconvolutionFunctor &) const
  {
    return true;
  }

  ITK_UNEQUAL_OPERATOR_MEMBER_FUNCTION(InverseDeconvolutionFunctor);

  inline TOutput
  operator()(const TInput1 & I, const TInput2 & H) const
  {
    const double absH = itk::Math::abs(H);
    TOutput      value{};
    if (absH >= m_KernelZeroMagnitudeThreshold)
    {
      value = static_cast<TOutput>(I / H);
    }
    return value;
  }

  /** Set/get the threshold value below which complex magnitudes are considered
   * to be zero. */
  /** @ITKStartGrouping */
  void
  SetKernelZeroMagnitudeThreshold(double mu)
  {
    m_KernelZeroMagnitudeThreshold = mu;
  }
  [[nodiscard]] double
  GetKernelZeroMagnitudeThreshold() const
  {
    return m_KernelZeroMagnitudeThreshold;
  }
  /** @ITKEndGrouping */
private:
  double m_KernelZeroMagnitudeThreshold{ 0.0 };
};
} // namespace Functor

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkInverseDeconvolutionImageFilter.hxx"
#endif

#endif
