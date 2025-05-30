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

#include <fstream>
#include "itkBoxSigmaImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTextOutput.h"
#include "itkSimpleFilterWatcher.h"
#include "itkTestingMacros.h"

int
itkBoxSigmaImageFilterTest(int argc, char * argv[])
{
  // Comment the following if you want to use the itk text output window
  itk::OutputWindow::SetInstance(itk::TextOutput::New());

  if (argc < 4)
  {
    std::cerr << "Usage: " << itkNameOfTestExecutableMacro(argv) << " InputImage BaselineImage radius" << std::endl;
    return EXIT_FAILURE;
  }

  using ImageType = itk::Image<unsigned char, 2>;

  using ReaderType = itk::ImageFileReader<ImageType>;
  auto input = ReaderType::New();
  input->SetFileName(argv[1]);

  // Create a filter
  using FilterType = itk::BoxSigmaImageFilter<ImageType, ImageType>;
  auto                           filter = FilterType::New();
  const itk::SimpleFilterWatcher filterWatch(filter);

  using RadiusType = FilterType::RadiusType;

  // test default values
  auto r1 = itk::MakeFilled<RadiusType>(1);
  if (filter->GetRadius() != r1)
  {
    std::cerr << "Wrong default Radius." << std::endl;
    return EXIT_FAILURE;
  }

  // set radius with a radius type
  auto r5 = itk::MakeFilled<RadiusType>(5);
  filter->SetRadius(r5);
  if (filter->GetRadius() != r5)
  {
    std::cerr << "Radius value is not the expected one: r5." << std::endl;
    return EXIT_FAILURE;
  }

  // set radius with an integer
  filter->SetRadius(1);
  if (filter->GetRadius() != r1)
  {
    std::cerr << "Radius value is not the expected one: r1." << std::endl;
    return EXIT_FAILURE;
  }

  try
  {
    const int r = std::stoi(argv[3]);
    filter->SetInput(input->GetOutput());
    filter->SetRadius(r);
    filter->Update();
  }
  catch (const itk::ExceptionObject & e)
  {
    std::cerr << "Exception detected: " << e.GetDescription();
    return EXIT_FAILURE;
  }

  // Generate test image
  using WriterType = itk::ImageFileWriter<ImageType>;
  auto writer = WriterType::New();
  writer->SetInput(filter->GetOutput());
  writer->SetFileName(argv[2]);
  writer->Update();

  return EXIT_SUCCESS;
}
